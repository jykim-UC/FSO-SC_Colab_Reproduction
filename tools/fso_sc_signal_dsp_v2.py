"""Received-sample-only rectangular-8QAM carrier acquisition repair.

Append this source after ``fso_sc_signal_dsp.py`` in a notebook.  Standalone
imports are also supported.  QPSK and 16QAM dispatch to the unmodified v1
receiver, and the existing SignalDSPConfig dataclass is not changed.

The fourth-moment periodogram is only a coarse frequency estimator: a
nonuniform finite payload can bias its peak even without receiver noise.
Rectangular 8QAM also has *data-dependent* phase aliases on constant-amplitude
runs; these are not the constellation's pi rotational symmetry.  Independent
BPS windows can therefore jump between aliases on zero-padded coded data.

The repair refines frequency AND constant phase by the AWGN constellation
likelihood, then uses windowed decision-directed phase maximum likelihood
inside the acquired basin.  For fixed decisions that phase estimate is
arg(sum(y * conj(decision))).  No source bits, channel phase, codec state,
parity checks, or image-quality metric enters this receiver.  The initial
phase-branch-near-zero assumption and known real block magnitude are retained
from v1; fundamental blind phase ambiguity is not claimed to be solved.

Related work: Yan et al., JLT 37, 5856-5866 (2019) document nonuniform-QAM
failure of conventional fourth-power frequency estimation.  Meiyappan et al.,
JLT 31, 2055-2069 (2013) study decision-aided ML phase/frequency recovery.
These motivate the estimator family, not a claim that this simple local
likelihood search reproduces either paper's particular algorithm.
"""

import math

import torch
import torch.nn.functional as F


if "recover_qam_carrier" not in globals():
    from fso_sc_signal_dsp import (
        SignalDSPConfig,
        _estimate_cfo_fourth_power,
        _unwrap_symmetry_phase,
        recover_qam_carrier,
    )
_rect8_legacy_recover_qam_carrier = globals().get(
    "_rect8_legacy_recover_qam_carrier", recover_qam_carrier
)


RECT8_DSP_VERSION = "rect8_constellation_ml_cfr_dd_ml_cpr_v2"
RECT8_DSP_REFERENCE = {
    "affected_modulation": "8QAM",
    "scope": "rectangular 8QAM only; QPSK and 16QAM use unchanged v1",
    "coarse_cfr": "v1 fourth-power FFT and Newton periodogram refinement",
    "fine_cfr": "received-sample constellation-distance likelihood; frequency and constant phase jointly fitted",
    "cpr": "two windowed decision-directed ML phase updates, arg(sum(y*conj(decision)))",
    "acquisition_search": "one unoversampled fourth-harmonic CFO bin either side of the coarse estimate",
    "acquisition_samples": "at most 2048 uniformly spaced received samples; no data-dependent sampling",
    "fine_frequency_trials": 17,
    "fine_newton_steps": 5,
    "phase_trials_and_window": "unchanged SignalDSPConfig phase count and window length",
    "phase_ambiguity": "initial phase branch nearest zero; unidentifiable likelihood ties use that convention",
    "no_pilots": True,
    "uses_transmitted_bits_or_symbols": False,
    "uses_true_channel_phase": False,
    "uses_decoder_or_codec_feedback": False,
    "amplitude_default": "known positive real block magnitude only, as in v1",
    "failure_policy": "estimates and errors retained; no frame rejection",
    "legacy_bps_reference": "https://doi.org/10.1109/JLT.2008.2010511",
    "nonuniform_qam_cfr_related_work": "https://opg.optica.org/jlt/abstract.cfm?uri=jlt-37-23-5856",
    "decision_aided_ml_related_work": "https://opg.optica.org/jlt/abstract.cfm?uri=jlt-31-13-2055",
    "related_work_scope": "estimator-family motivation only; neither paper's exact algorithm is reproduced",
}


def rect8_repair_metadata():
    """JSON-safe provenance for the notebook's additional receiver stage."""
    return {"receiver_version": RECT8_DSP_VERSION, **RECT8_DSP_REFERENCE}


def _rect8_nearest_point(value):
    """Exact nearest point for I={-3,-1,1,3}/sqrt(6), Q={-1,1}/sqrt(6)."""
    root = math.sqrt(6.0)
    i = (2 * ((value.real * root + 3) / 2).round().clamp(0, 3) - 3) / root
    q = (2 * (value.imag >= 0).to(value.real.dtype) - 1) / root
    return torch.complex(i, q)


def _rect8_joint_likelihood_acquisition(observed, symbol_rate_baud, config, fit_frequency):
    """Local frequency acquisition with continuous constant-phase refinement.

    A fourth-harmonic Fourier bin corresponds to Fs/(4*N) carrier Hz, or
    pi/2 radians of end-to-end carrier phase.  This fixes the search width
    from the acquisition resolution, not from any image/BER experiment.
    Equally likely phase solutions are resolved by the declared initial
    near-zero phase convention, with a floating-point roundoff tolerance.
    """
    batch, count = observed.shape
    dtype, device = observed.real.dtype, observed.device
    sample_indices = torch.linspace(0, count - 1, min(count, 2048), device=device).round().long()
    sample_time = sample_indices.to(dtype) / (count - 1) - 0.5
    initial_phases = (torch.arange(config.bps_test_phases, device=device, dtype=dtype)
                      / config.bps_test_phases - 0.5) * math.pi
    slopes = (torch.linspace(-math.pi / 2, math.pi / 2, 17, device=device, dtype=dtype)
              if fit_frequency else torch.zeros(1, device=device, dtype=dtype))
    fitted_hz, fitted_intercept, fitted_cost = [], [], []
    for row in observed:
        sample = row[sample_indices]
        scale = sample.abs().amax().clamp_min(1.0)
        phase_candidates, slope_candidates = [], []
        # Bound temporary memory independently of frame/batch length.
        for phase_chunk in initial_phases.split(16):
            angle = phase_chunk[:, None, None] + slopes[None, :, None] * sample_time
            rotated = sample * torch.exp(-1j * angle)
            costs = ((rotated - _rect8_nearest_point(rotated)) / scale).abs().square().mean(-1)
            phase_candidates.append(phase_chunk)
            slope_candidates.append(slopes[costs.argmin(-1)])
        phase = torch.cat(phase_candidates)
        slope = torch.cat(slope_candidates)
        # Newton optimization for the local piecewise-smooth likelihood.
        # Keeping all phase seeds also retains data-dependent ambiguous basins.
        for _ in range(5):
            rotated = sample * torch.exp(-1j * (phase[:, None] + slope[:, None] * sample_time))
            decisions = _rect8_nearest_point(rotated)
            cross = (rotated / scale) * decisions.conj()
            h00 = cross.real.mean(-1)
            g0 = cross.imag.mean(-1)
            if fit_frequency:
                h01 = (cross.real * sample_time).mean(-1)
                h11 = (cross.real * sample_time.square()).mean(-1)
                g1 = (cross.imag * sample_time).mean(-1)
                determinant = h00 * h11 - h01.square()
                stable = (h00 > 0) & (determinant > torch.finfo(dtype).eps * (h00 * h11).abs())
                denominator = torch.where(stable, determinant, torch.ones_like(determinant))
                phase_step = torch.where(stable, (h11 * g0 - h01 * g1) / denominator, 0.0)
                slope_step = torch.where(stable, (h00 * g1 - h01 * g0) / denominator, 0.0)
            else:
                stable = h00 > torch.finfo(dtype).tiny
                phase_step = torch.where(stable, g0 / h00.clamp_min(torch.finfo(dtype).tiny), 0.0)
                slope_step = torch.zeros_like(slope)
            phase = phase + phase_step.clamp(-math.pi / 8, math.pi / 8)
            slope = (slope + slope_step.clamp(-math.pi / 4, math.pi / 4)).clamp(-math.pi / 2, math.pi / 2)
        rotated = sample * torch.exp(-1j * (phase[:, None] + slope[:, None] * sample_time))
        costs = ((rotated - _rect8_nearest_point(rotated)) / scale).abs().square().mean(-1)
        intercept = torch.remainder(phase - slope / 2 + math.pi / 2, math.pi) - math.pi / 2
        tolerance = 32 * torch.finfo(dtype).eps
        eligible = costs <= costs.amin() + tolerance
        selection = torch.where(eligible, intercept.abs(), float("inf")).argmin()
        fitted_hz.append(slope[selection] * symbol_rate_baud / (2 * math.pi * (count - 1)))
        fitted_intercept.append(intercept[selection])
        fitted_cost.append(costs[selection])
    frequency = torch.stack(fitted_hz)
    intercept = torch.stack(fitted_intercept)
    index = torch.arange(count, device=device, dtype=dtype)
    phase = intercept[:, None] + frequency[:, None] * (2 * math.pi / symbol_rate_baud) * index
    return observed * torch.exp(-1j * phase), frequency, intercept, torch.stack(fitted_cost)


def _rect8_decision_directed_phase(observed, config):
    """Windowed constant-phase ML, initialized inside the acquired basin."""
    corrected = observed
    scale = observed.abs().amax(-1, keepdim=True).clamp_min(1.0)
    phase = torch.zeros_like(observed.real)
    unwrap_events = torch.zeros(observed.shape[0], device=observed.device, dtype=torch.int64)
    for _ in range(2):
        decisions = _rect8_nearest_point(corrected)
        cross = (observed / scale) * decisions.conj()
        kwargs = dict(kernel_size=config.bps_window_symbols, stride=1,
                      padding=config.bps_window_symbols // 2, count_include_pad=False)
        average = torch.complex(F.avg_pool1d(cross.real[:, None], **kwargs)[:, 0],
                                F.avg_pool1d(cross.imag[:, None], **kwargs)[:, 0])
        phase, unwrap_events = _unwrap_symmetry_phase(average.angle(), 2 * math.pi)
        corrected = observed * torch.exp(-1j * phase)
    return corrected, phase, unwrap_events


@torch.no_grad()
def recover_qam_carrier(received, modulation, symbol_rate_baud, config=None, known_block_magnitude=None):
    """V1-compatible API; only rectangular-8QAM takes the repaired path."""
    if modulation.upper() != "8QAM":
        return _rect8_legacy_recover_qam_carrier(
            received, modulation, symbol_rate_baud, config, known_block_magnitude
        )
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
        amplitude = received.abs().square().mean(-1, keepdim=True).sqrt()
        amplitude = amplitude.clamp_min(torch.finfo(received.real.dtype).tiny)
    observed = received / amplitude
    zeros = torch.zeros(received.shape[0], device=received.device, dtype=received.real.dtype)
    if config.cfo_enabled:
        observed, coarse_hz, peak_ratio = _estimate_cfo_fourth_power(observed, symbol_rate_baud, config)
    else:
        coarse_hz, peak_ratio = zeros, zeros
    if config.cfo_enabled or config.cpr_enabled:
        acquired, fine_hz, initial_phase, cost = _rect8_joint_likelihood_acquisition(
            observed, symbol_rate_baud, config, config.cfo_enabled
        )
        if config.cpr_enabled:
            observed = acquired
        else:
            index = torch.arange(received.shape[-1], device=received.device, dtype=received.real.dtype)
            observed = observed * torch.exp(-1j * fine_hz[:, None] * (2 * math.pi / symbol_rate_baud) * index)
    else:
        fine_hz, initial_phase, cost = zeros, zeros, zeros
    if config.cpr_enabled:
        observed, phase, unwrap_events = _rect8_decision_directed_phase(observed, config)
        phase = phase + initial_phase[:, None]
    else:
        phase = torch.zeros_like(received.real)
        unwrap_events = zeros.to(torch.int64)
    diagnostics = {
        "receiver_version": RECT8_DSP_VERSION,
        "amplitude_mode": config.amplitude_mode,
        "estimated_cfo_hz": (coarse_hz + fine_hz).detach().cpu().tolist(),
        "coarse_cfo_hz": coarse_hz.detach().cpu().tolist(),
        "fine_cfo_correction_hz": fine_hz.detach().cpu().tolist(),
        "acquisition_normalized_distance": cost.detach().cpu().tolist(),
        "cfr_peak_to_mean_power": peak_ratio.detach().cpu().tolist(),
        "cfr_unambiguous_interval_hz": [-symbol_rate_baud / 8, symbol_rate_baud / 8],
        "estimated_phase_start_rad": phase[:, 0].detach().cpu().tolist(),
        "estimated_phase_end_rad": phase[:, -1].detach().cpu().tolist(),
        "phase_symmetry_unwrap_events": unwrap_events.detach().cpu().tolist(),
        "failure_policy": "estimates/errors retained, no frame rejection",
    }
    return observed, diagnostics


def rect8_repair_self_test(device="cpu"):
    """Small synthetic preflight; no model, image, codec, or external data."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(20261028)
    count, sample_rate = 7168, 28e9
    ids = torch.randint(8, (1, count), generator=generator, device=device)
    ids[:, 5000:] = 0
    i = torch.tensor([-3, -3, -1, -1, 3, 3, 1, 1], device=device) / math.sqrt(6)
    q = torch.tensor([-1, 1, -1, 1, -1, 1, -1, 1], device=device) / math.sqrt(6)
    points = torch.complex(i, q)
    source = points[ids]
    index = torch.arange(count, device=device, dtype=source.real.dtype)
    records = []
    for cfo in (0.0, 700e6):
        received = source * torch.exp(1j * (0.03 + 2 * math.pi * cfo / sample_rate * index))
        recovered, diagnostics = recover_qam_carrier(
            received, "8QAM", sample_rate, known_block_magnitude=torch.ones((1, 1), device=device)
        )
        errors = int(((recovered[..., None] - points).abs().argmin(-1) != ids).sum())
        if errors:
            raise AssertionError(("rect8 preflight failed", cfo, errors, diagnostics))
        records.append({"cfo_hz": cfo, "symbol_errors": errors,
                        "estimated_cfo_hz": diagnostics["estimated_cfo_hz"],
                        "acquisition_normalized_distance": diagnostics["acquisition_normalized_distance"]})
    return {"receiver_version": RECT8_DSP_VERSION, "synthetic_only": True,
            "symbols_per_condition": count, "zero_tail_symbols": 2168, "conditions": records}
