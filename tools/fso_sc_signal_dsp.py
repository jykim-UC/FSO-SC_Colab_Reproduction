"""Signal-estimated symbol-rate coherent carrier recovery for the FSO notebook.

Embed after ``qam_index_link`` and ``paper_fso_channel_v1``. This is a
plausible receiver implementation, NOT the undisclosed VPI DSP used by Chen
et al. It introduces no pilot symbols and never uses transmitted symbol values
or the simulated channel phase to recover the carrier. The default magnitude
mode retains explicitly assumed block-amplitude CSI; ``blind_rms`` is also
available, but its uniform-symbol assumption is unsuitable for strongly biased
semantic bitstreams.

Reference for feed-forward blind phase search: Pfau, Hoffmann, and Noe,
JLT 27(8), 989-999 (2009), doi:10.1109/JLT.2008.2010511. Fourth-power spectral
CFR is a conventional modulation-removal approximation, with its finite-frame
failure modes retained. Exact frame/timing/IQ calibration and an initial phase
branch near zero remain assumptions. No RRC/ADC waveform is synthesized here.
"""

from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F


SIGNAL_DSP_VERSION = "fourth_power_fft_newton_bps_v1"
SIGNAL_DSP_REFERENCE = {
    "blind_phase_search": "https://doi.org/10.1109/JLT.2008.2010511",
    "scope": "symbol-rate, single receiver; no waveform/timing/IQ calibration model",
    "cfr": "fourth-power FFT peak plus continuous periodogram Newton refinement",
    "cpr": "windowed minimum-distance blind phase search and symmetry unwrapping",
    "phase_ambiguity": (
        "initial carrier phase lies in the branch nearest zero, consistent with "
        "the simulator phase origin; no transmitted data is used to select a branch"
    ),
    "no_pilots": True,
    "amplitude_default": "known block magnitude only, not the complex channel gain",
    "failure_policy": "keep estimates/errors; do not drop low-SNR frames",
}


@dataclass(frozen=True)
class SignalDSPConfig:
    amplitude_mode: str = "known_block_magnitude"
    cfo_fft_oversampling: int = 4
    cfo_newton_steps: int = 4
    bps_test_phases: int = 64
    bps_window_symbols: int = 65
    cfo_enabled: bool = True
    cpr_enabled: bool = True

    def __post_init__(self):
        if self.amplitude_mode not in {"known_block_magnitude", "blind_rms"}:
            raise ValueError("Unsupported amplitude_mode")
        if self.cfo_fft_oversampling < 1 or self.cfo_newton_steps < 0:
            raise ValueError("Invalid CFR FFT/refinement configuration")
        if self.bps_test_phases < 8:
            raise ValueError("Use at least eight BPS trial phases")
        if self.bps_window_symbols < 1 or self.bps_window_symbols % 2 != 1:
            raise ValueError("BPS window must be a positive odd integer")


def _qam_axis_levels(modulation, device, dtype):
    modulation = modulation.upper()
    if modulation == "QPSK":
        i = q = torch.tensor([-1.0, 1.0], device=device, dtype=dtype) / math.sqrt(2)
        period = math.pi / 2
    elif modulation == "16QAM":
        i = q = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device, dtype=dtype) / math.sqrt(10)
        period = math.pi / 2
    elif modulation == "8QAM":
        # Rectangular geometry in qam_constellation, NOT square-QAM symmetry.
        i = torch.tensor([-3.0, -1.0, 1.0, 3.0], device=device, dtype=dtype) / math.sqrt(6)
        q = torch.tensor([-1.0, 1.0], device=device, dtype=dtype) / math.sqrt(6)
        period = math.pi
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return i, q, period


def _estimate_cfo_fourth_power(observed, symbol_rate_baud, config):
    batch, count = observed.shape
    # The fourth harmonic repeats at Fs/4, so the unambiguous CFO interval is
    # [-Fs/8, Fs/8). Paper residual offsets are within that acquisition range.
    nfft = 1 << max(1, (count * config.cfo_fft_oversampling - 1).bit_length())
    # Severe fog can make gain-normalized receiver noise enormous. Remove a
    # positive block scale before the fourth power; it does not change the
    # frequency peak and prevents overflow in frames whose data is unrecoverable.
    peak_scale = observed.abs().amax(dim=-1, keepdim=True).clamp_min(
        torch.finfo(observed.real.dtype).tiny
    )
    cfr_samples = observed / peak_scale
    fourth = cfr_samples.square().square()
    spectrum = torch.fft.fft(fourth, n=nfft, dim=-1)
    power = spectrum.abs().square()
    peak = power.argmax(dim=-1)
    signed_peak = torch.where(peak > nfft // 2, peak - nfft, peak)
    real_dtype = observed.real.dtype
    omega = signed_peak.to(real_dtype) * (2.0 * math.pi / nfft)
    index = torch.arange(count, device=observed.device, dtype=real_dtype)
    centered = index - (count - 1) / 2.0
    max_step = 2.0 * math.pi / nfft
    for _ in range(config.cfo_newton_steps):
        derotated = fourth * torch.exp(-1j * omega[:, None] * centered[None, :])
        value = derotated.sum(dim=-1)
        derivative = (-1j * centered[None, :] * derotated).sum(dim=-1)
        curvature = (-centered.square()[None, :] * derotated).sum(dim=-1)
        slope = 2 * (derivative * value.conj()).real
        second = 2 * ((curvature * value.conj()).real + derivative.abs().square())
        stable = second < -torch.finfo(real_dtype).eps
        denominator = torch.where(stable, second, torch.ones_like(second))
        step = torch.where(stable, slope / denominator, torch.zeros_like(slope))
        omega = omega - step.clamp(min=-max_step, max=max_step)
    omega = torch.remainder(omega + math.pi, 2 * math.pi) - math.pi
    estimated_hz = omega * symbol_rate_baud / (8 * math.pi)
    phase = omega[:, None] * index[None, :] / 4.0
    compensated = observed * torch.exp(-1j * phase)
    peak_ratio = power.max(dim=-1).values / power.mean(dim=-1).clamp_min(
        torch.finfo(real_dtype).tiny
    )
    return compensated, estimated_hz, peak_ratio


def _unwrap_symmetry_phase(phase, period):
    differences = torch.diff(phase, dim=-1)
    wrapped = torch.remainder(differences + period / 2, period) - period / 2
    corrections = wrapped - differences
    cumulative = torch.cat(
        [torch.zeros_like(phase[:, :1]), corrections.cumsum(dim=-1)], dim=-1
    )
    return phase + cumulative, (corrections.abs() > period / 2).sum(dim=-1)


def _blind_phase_search(observed, modulation, config):
    i_levels, q_levels, period = _qam_axis_levels(
        modulation, observed.device, observed.real.dtype
    )
    trial_phases = (
        torch.arange(config.bps_test_phases, device=observed.device, dtype=observed.real.dtype)
        / config.bps_test_phases - 0.5
    ) * period
    rotated = observed[:, None, :] * torch.exp(-1j * trial_phases[None, :, None])
    i_distance = (rotated.real[..., None] - i_levels).square().amin(dim=-1)
    q_distance = (rotated.imag[..., None] - q_levels).square().amin(dim=-1)
    costs = F.avg_pool1d(
        i_distance + q_distance,
        kernel_size=config.bps_window_symbols,
        stride=1,
        padding=config.bps_window_symbols // 2,
        count_include_pad=False,
    )
    selected = trial_phases[costs.argmin(dim=1)]
    phase, unwrap_events = _unwrap_symmetry_phase(selected, period)
    # Global rotational ambiguity is not solved using the source symbols. The
    # initial branch near zero is an explicit simulator convention, not a pilot.
    phase = phase - torch.round(phase[:, :1] / period) * period
    corrected = observed * torch.exp(-1j * phase)
    return corrected, phase, unwrap_events


@torch.no_grad()
def recover_qam_carrier(
    received,
    modulation,
    symbol_rate_baud,
    config=None,
    known_block_magnitude=None,
):
    """Estimate CFR/CPR from received samples, without source data/true phase.

    ``known_block_magnitude`` may contain positive block-amplitude CSI only.
    It must not contain the true phase trajectory or the transmitted symbols.
    """
    config = SignalDSPConfig() if config is None else config
    if received.ndim != 2 or not torch.is_complex(received) or received.shape[-1] < 8:
        raise ValueError("Expected complex [batch, symbols>=8] receiver samples")
    if symbol_rate_baud <= 0 or not torch.isfinite(received).all():
        raise ValueError("Invalid receiver samples or symbol rate")
    if config.amplitude_mode == "known_block_magnitude":
        if known_block_magnitude is None or torch.is_complex(known_block_magnitude):
            raise ValueError("Positive REAL block-amplitude CSI is required")
        if not torch.isfinite(known_block_magnitude).all() or (known_block_magnitude <= 0).any():
            raise ValueError("Invalid block-amplitude CSI")
        amplitude = torch.broadcast_to(known_block_magnitude, (received.shape[0], 1))
    else:
        amplitude = received.abs().square().mean(dim=-1, keepdim=True).sqrt()
        amplitude = amplitude.clamp_min(torch.finfo(received.real.dtype).tiny)
    observed = received / amplitude
    zeros = torch.zeros(received.shape[0], device=received.device, dtype=received.real.dtype)
    if config.cfo_enabled:
        observed, cfo_hz, peak_ratio = _estimate_cfo_fourth_power(
            observed, symbol_rate_baud, config
        )
    else:
        cfo_hz, peak_ratio = zeros, zeros
    if config.cpr_enabled:
        observed, phase, unwrap_events = _blind_phase_search(observed, modulation, config)
    else:
        phase = torch.zeros_like(received.real)
        unwrap_events = zeros.to(torch.int64)
    diagnostics = {
        "receiver_version": SIGNAL_DSP_VERSION,
        "amplitude_mode": config.amplitude_mode,
        "estimated_cfo_hz": cfo_hz.detach().cpu().tolist(),
        "cfr_peak_to_mean_power": peak_ratio.detach().cpu().tolist(),
        "cfr_unambiguous_interval_hz": [-symbol_rate_baud / 8, symbol_rate_baud / 8],
        "estimated_phase_start_rad": phase[:, 0].detach().cpu().tolist(),
        "estimated_phase_end_rad": phase[:, -1].detach().cpu().tolist(),
        "phase_symmetry_unwrap_events": unwrap_events.detach().cpu().tolist(),
        "failure_policy": "estimates/errors retained, no frame rejection",
    }
    return observed, diagnostics


@dataclass
class SignalDSPFSOReceiver:
    """Replace phase-oracle equalization for one physical receiver branch."""

    physical_receiver: object
    modulation: str = "QPSK"
    dsp_config: object = None

    def __post_init__(self):
        if self.dsp_config is None:
            self.dsp_config = SignalDSPConfig()
        if self.physical_receiver.config.num_receivers != 1:
            raise ValueError("Signal DSP wrapper supports single Rx only; no oracle MRC")

    def reproducibility_config(self):
        return {
            "physical_receiver": self.physical_receiver.reproducibility_config(),
            "receiver_version": SIGNAL_DSP_VERSION,
            "receiver_config": asdict(self.dsp_config),
            "receiver_assumptions": SIGNAL_DSP_REFERENCE,
            "modulation": self.modulation.upper(),
        }

    @torch.no_grad()
    def __call__(self, symbols, **channel_kwargs):
        observation = self.physical_receiver(symbols, **channel_kwargs)
        if observation.is_equalized:
            raise ValueError("Expected un-equalized physical receiver observation")
        magnitude = None
        if self.dsp_config.amplitude_mode == "known_block_magnitude":
            # Deliberately read only real block magnitude, NOT its phase. Do
            # not pass complex channel_gain to recover_qam_carrier.
            magnitude = observation.channel_gain.abs().mean(dim=-1, keepdim=True)
        recovered, receiver_diagnostics = recover_qam_carrier(
            observation.symbols,
            self.modulation,
            self.physical_receiver.config.symbol_rate_baud,
            self.dsp_config,
            known_block_magnitude=magnitude,
        )
        diagnostics = dict(observation.diagnostics or {})
        diagnostics["signal_dsp"] = receiver_diagnostics
        return FSOChannelOutput(symbols=recovered, is_equalized=True, diagnostics=diagnostics)


@dataclass
class SignalDSPPaperIndexChannel:
    """Drop-in single-Rx alternative to PaperFSOIndexChannel in this notebook."""

    receiver: object
    dsp_config: object = None
    modulations: tuple = ("QPSK", "8QAM", "16QAM")
    bits_per_index: int = 14

    def __post_init__(self):
        if self.dsp_config is None:
            self.dsp_config = SignalDSPConfig()
        if self.receiver.config.num_receivers != 1:
            raise ValueError("Estimated carrier receiver is currently single-Rx only")

    @property
    def receiver_config(self):
        return {
            "physical_receiver": self.receiver.reproducibility_config(),
            "receiver_version": SIGNAL_DSP_VERSION,
            "receiver_config": asdict(self.dsp_config),
            "receiver_assumptions": SIGNAL_DSP_REFERENCE,
        }

    def reproducibility_config(self):
        return {
            "qam_mapping_version": QAM_MAPPING_VERSION,
            "modulations": tuple(name.upper() for name in self.modulations),
            "bits_per_index": self.bits_per_index,
            "receiver": self.receiver_config,
        }

    def __call__(self, indices, modulation="QPSK", return_statistics=False, **channel_kwargs):
        modulation = modulation.upper()
        if modulation not in tuple(name.upper() for name in self.modulations):
            raise ValueError(f"Unsupported modulation: {modulation}")
        receiver = SignalDSPFSOReceiver(self.receiver, modulation, self.dsp_config)
        link = QAMIndexLink(
            receiver,
            modulation=modulation,
            bits_per_index=self.bits_per_index,
            receiver_config=receiver.reproducibility_config(),
        )
        return link(indices, return_statistics=return_statistics, **channel_kwargs)


def signal_dsp_offline_self_test(device="cpu"):
    """Synthetic local checks only; does not download weights or run training."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(71819)
    count = 4096
    sample_rate = 28e9
    time = torch.arange(count, device=device) / sample_rate
    records = {}
    for modulation in ("QPSK", "8QAM", "16QAM"):
        points, _, _ = qam_constellation(modulation, device)
        ids = torch.randint(len(points), (2, count), generator=generator, device=device)
        source = points[ids]
        true_cfo = torch.tensor([700e6, -320e6], device=device)
        phase = 0.03 + 2 * math.pi * true_cfo[:, None] * time[None, :]
        amplitude = torch.tensor([[0.002], [0.0005]], device=device)
        noise = torch.complex(
            torch.randn(source.shape, generator=generator, device=device),
            torch.randn(source.shape, generator=generator, device=device),
        ) * (0.005 / math.sqrt(2))
        received = amplitude * (source * torch.exp(1j * phase) + noise)
        recovered, stats = recover_qam_carrier(
            received, modulation, sample_rate, known_block_magnitude=amplitude
        )
        recovered_ids = (recovered[..., None] - points).abs().square().argmin(dim=-1)
        errors = int((recovered_ids != ids).sum())
        estimated_cfo = torch.tensor(stats["estimated_cfo_hz"], device=device)
        assert errors == 0, (modulation, errors, stats)
        # Finite-frame fourth-moment self-noise is nonzero for QAM. Require
        # error below one 4th-harmonic FFT bin; BPS must still give exact data.
        assert (estimated_cfo - true_cfo).abs().max() < sample_rate / (4 * count), stats
        records[modulation] = {"symbol_errors": errors, "estimated_cfo_hz": stats["estimated_cfo_hz"]}

    # Nonzero phase trajectory must be inferred and unwrapped, not divided out
    # with a hidden simulator phase. Deliberately turn CFR off for this check.
    points, _, _ = qam_constellation("QPSK", device)
    ids = torch.randint(len(points), (1, count), generator=generator, device=device)
    source = points[ids]
    phase = torch.linspace(0, 3, count, device=device)[None, :]
    recovered, stats = recover_qam_carrier(
        source * torch.exp(1j * phase), "QPSK", sample_rate,
        SignalDSPConfig(cfo_enabled=False), torch.ones((1, 1), device=device),
    )
    recovered_ids = (recovered[..., None] - points).abs().square().argmin(dim=-1)
    assert torch.equal(ids, recovered_ids), stats
    assert stats["phase_symmetry_unwrap_events"][0] >= 1

    # A global pi/2 ambiguity is intentionally NOT resolved using source data.
    received = source * torch.exp(torch.tensor(1j * math.pi / 2, device=device))
    recovered, _ = recover_qam_carrier(
        received, "QPSK", sample_rate,
        known_block_magnitude=torch.ones((1, 1), device=device),
    )
    recovered_ids = (recovered[..., None] - points).abs().square().argmin(dim=-1)
    assert not torch.equal(ids, recovered_ids), "Undisclosed source-aided phase ambiguity resolution"
    print("Signal-estimated CFR/CPR offline self-test passed:", records)
    return records
