"""Experimental QPSK block-likelihood acquisition, not author VPI DSP.

V1/V2 are preserved.  This candidate replaces only QPSK when called explicitly.
It uses received samples and the existing real block-amplitude CSI, never source
bits, true phase/CFO, decoder feedback or image scores.  A global residual-CFO
bound is derived from the configured orbit and wavelength, NOT the actual angle
or simulated CFO.  No frame is dropped.  The zero-frequency grid candidate is
always included, alongside received-spectrum candidates, and is never forced.

The short frames at 28 GBaud / 1 kHz linewidth motivate one affine phase per
frame instead of unwrapping independent 65-symbol noisy BPS windows.  This is
an explicit slow-phase assumption and can fail for fast phase wander.  Blind
QPSK's pi/2 ambiguity is retained; nearest-zero initial branch is an assumption.

References: Viterbi & Viterbi, IEEE Trans. IT 29(4), 543-551 (1983),
doi:10.1109/TIT.1983.1056713 (nonlinear M-PSK phase estimation); OptiCommPy
fourthPowerFOE documentation for M-th power spectral acquisition.  The exact
finite candidate search below is an independent implementation, not either
publication's released algorithm and not a confirmed FSO-SC author setting.
"""
from dataclasses import asdict, dataclass
import math

import torch
import torch.nn.functional as F

if "_rect8_legacy_recover_qam_carrier" not in globals():
    from fso_sc_signal_dsp_v2 import recover_qam_carrier as _v3_previous_recover
    from fso_sc_signal_dsp import SignalDSPConfig
else:
    _v3_previous_recover = recover_qam_carrier


QPSK_BLOCK_DSP_VERSION = "qpsk_bounded_spectral_marginal_ml_block_v3_candidate"


@dataclass(frozen=True)
class QPSKBlockMLConfig:
    wavelength_m: float = 1550e-9
    orbital_radius_m: float = 6371e3 + 620e3
    gravitational_parameter: float = 3.986004418e14
    residual_doppler_ratio: float = .1
    spectral_peaks_per_statistic: int = 8
    phase_trials: int = 16
    newton_steps: int = 6
    maximum_fit_samples: int = 8192

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in (self.wavelength_m,
                self.orbital_radius_m,self.gravitational_parameter,self.residual_doppler_ratio)):
            raise ValueError("Positive finite orbital/CFO parameters required")
        if any(type(value) is not int for value in (self.spectral_peaks_per_statistic,
                self.phase_trials,self.newton_steps,self.maximum_fit_samples)):
            raise ValueError("Integer QPSK search parameters required")
        if (self.spectral_peaks_per_statistic < 1 or self.phase_trials < 8
                or self.newton_steps < 1 or self.maximum_fit_samples < 128):
            raise ValueError("Invalid QPSK block-ML search configuration")

    @property
    def maximum_residual_cfo_hz(self):
        return self.residual_doppler_ratio * math.sqrt(
            self.gravitational_parameter/self.orbital_radius_m)/self.wavelength_m


def qpsk_block_ml_metadata(config=None):
    config = QPSKBlockMLConfig() if config is None else config
    return {"receiver_version":QPSK_BLOCK_DSP_VERSION,"experimental_candidate":True,
        "scope":"QPSK only; 8QAM/16QAM unchanged v2/v1", "config":asdict(config),
        "maximum_residual_cfo_hz":config.maximum_residual_cfo_hz,
        "frequency_prior":"global circular-orbit speed / wavelength times residual ratio; no actual CFO/angle",
        "phase_model":"one affine phase across a short frame; no local BPS unwrapping",
        "likelihood":"uniform-QPSK symbol-marginal AWGN likelihood; no transmitted decisions",
        "noise_variance":"normalized frame energy minus unit QPSK signal power, floor 1e-3",
        "acquisition_gate":"retain nominal zero residual CFO unless N*loglikelihood gain exceeds .5*log(N)+log(number of resolvable global CFO bins); conservative independent implementation choice",
        "nominal_cfo_prior":"zero after nominal carrier/precompensation, never actual simulated CFO; nonzero weak signals may fail acquisition",
        "phase_ambiguity":"initial branch nearest zero, not blind ambiguity resolution",
        "uses_true_channel_phase":False,"uses_true_cfo":False,
        "uses_transmitted_bits_or_symbols":False,"uses_codec_or_lpips_feedback":False,
        "no_pilots":True,"failure_policy":"all estimated frames and errors retained",
        "references":["https://doi.org/10.1109/TIT.1983.1056713",
        "https://opticommpy.readthedocs.io/en/stable/generated/optic.dsp.carrierRecovery.fourthPowerFOE.html"]}


def _qpsk_candidates(row, rate, config):
    count=row.numel(); nfft=1 << (4*count-1).bit_length()
    frequencies=torch.fft.fftfreq(nfft,1/rate,device=row.device)/4
    eligible=frequencies.abs() <= config.maximum_residual_cfo_hz
    indices=eligible.nonzero().flatten()
    candidates=[torch.zeros(1,device=row.device,dtype=row.real.dtype)]
    # Raw fourth moment and bounded-amplitude fourth moment.  The latter limits
    # the influence of rare AWGN amplitude outliers without selecting symbols.
    energy=row.abs().square().mean().clamp_min(torch.finfo(row.real.dtype).tiny)
    for value in (row, row/(row.abs().square()+energy).sqrt()):
        scale=value.abs().amax().clamp_min(torch.finfo(row.real.dtype).tiny)
        fourth=(value/scale).square().square()
        power=torch.fft.fft(fourth,nfft).abs().square()
        k=min(config.spectral_peaks_per_statistic,indices.numel())
        selected=indices[power[indices].topk(k).indices]
        candidates.append(frequencies[selected])
    return torch.unique(torch.cat(candidates)), rate/(4*count)


def _qpsk_fit(row, rate, config, fit_cfo=True):
    count=row.numel();dtype=row.real.dtype;device=row.device
    frequencies,bin_width=_qpsk_candidates(row,rate,config) if fit_cfo else (
        torch.zeros(1,device=device,dtype=dtype),rate/(4*count))
    ids=torch.linspace(0,count-1,min(count,config.maximum_fit_samples),device=device).round().long()
    time=ids.to(dtype)/(count-1)-.5
    sample=row[ids]
    noise=(row.abs().square().mean()-1).clamp_min(1e-3)
    c=math.sqrt(2)/noise
    phase_grid=(torch.arange(config.phase_trials,device=device,dtype=dtype)
                /config.phase_trials-.5)*(math.pi/2)
    base_frequency=frequencies.repeat_interleave(config.phase_trials)
    phase=phase_grid.repeat(frequencies.numel())
    slope=torch.zeros_like(phase)
    base_angle=base_frequency[:,None]*(2*math.pi/rate)*ids.to(dtype)
    for _ in range(config.newton_steps):
        rotated=sample*torch.exp(-1j*(base_angle+phase[:,None]+slope[:,None]*time))
        x,y=rotated.real,rotated.imag
        tx,ty=torch.tanh(c*x),torch.tanh(c*y)
        g=c*(y*tx-x*ty)
        h=c.square()*(y.square()*(1-tx.square())+x.square()*(1-ty.square()))-c*(x*tx+y*ty)
        h00=h.mean(-1);h01=(h*time).mean(-1);h11=(h*time.square()).mean(-1)
        g0=g.mean(-1);g1=(g*time).mean(-1)
        determinant=h00*h11-h01.square()
        stable=(h00 < 0)&(h11 < 0)&(determinant > 1e-8*(h00*h11).abs())
        denominator=torch.where(stable,determinant,torch.ones_like(determinant))
        phase_step=(torch.where(stable,(h11*g0-h01*g1)/denominator,0) if fit_cfo else
                    torch.where(h00<0,g0/h00.clamp_max(-torch.finfo(dtype).tiny),0))
        slope_step=torch.where(stable,(h00*g1-h01*g0)/denominator,0) if fit_cfo else torch.zeros_like(slope)
        phase=phase-phase_step.clamp(-math.pi/8,math.pi/8)
        slope=(slope-slope_step.clamp(-math.pi/4,math.pi/4)).clamp(-math.pi/2,math.pi/2)
    rotated=sample*torch.exp(-1j*(base_angle+phase[:,None]+slope[:,None]*time))
    logcosh=lambda x:F.softplus(2*x)-x-math.log(2)
    scores=(logcosh(c*rotated.real)+logcosh(c*rotated.imag)).mean(-1)
    frequency=base_frequency+slope*rate/(2*math.pi*(count-1))
    allowed=frequency.abs() <= config.maximum_residual_cfo_hz
    scores=torch.where(allowed,scores,-float("inf"))
    selected=scores.argmax()
    initial=phase[selected]-slope[selected]/2
    initial=torch.remainder(initial+math.pi/4,math.pi/2)-math.pi/4
    freq=frequency[selected]
    index=torch.arange(count,device=device,dtype=dtype)
    trajectory=initial+(2*math.pi/rate)*freq*index
    return row*torch.exp(-1j*trajectory),freq,initial,noise,scores[selected]


@torch.no_grad()
def recover_qam_carrier_v3(received,modulation,symbol_rate_baud,config=None,
                           known_block_magnitude=None,*,block_config=None):
    if modulation.upper() != "QPSK":
        return _v3_previous_recover(received,modulation,symbol_rate_baud,config,known_block_magnitude)
    config=SignalDSPConfig() if config is None else config
    block_config=QPSKBlockMLConfig() if block_config is None else block_config
    if received.ndim != 2 or not torch.is_complex(received) or received.shape[-1]<8:
        raise ValueError("Expected complex [batch,symbols>=8] receiver samples")
    if not math.isfinite(symbol_rate_baud) or symbol_rate_baud<=0 or not torch.isfinite(received).all():
        raise ValueError("Invalid receiver samples/rate")
    if block_config.maximum_residual_cfo_hz >= symbol_rate_baud/8:
        raise ValueError("Orbital CFO prior exceeds the fourth-power acquisition range")
    if config.amplitude_mode != "known_block_magnitude":
        raise ValueError("QPSK marginal likelihood requires declared real block-amplitude CSI")
    if known_block_magnitude is None or torch.is_complex(known_block_magnitude):
        raise ValueError("Positive real amplitude CSI required")
    if not torch.isfinite(known_block_magnitude).all() or (known_block_magnitude<=0).any():
        raise ValueError("Invalid amplitude CSI")
    amplitude=torch.broadcast_to(known_block_magnitude,(received.shape[0],1))
    observed=received/amplitude
    if not config.cfo_enabled and not config.cpr_enabled:
        return observed,{"receiver_version":QPSK_BLOCK_DSP_VERSION,"both_stages_disabled":True}
    rows=[];cfo=[];phase=[];noise=[];score=[];gates=[];gains=[];thresholds=[]
    for row in observed:
        recovered,f,p,n,s=_qpsk_fit(row,symbol_rate_baud,block_config,config.cfo_enabled)
        nominal_used=False; gain=0.;threshold=0.
        if config.cfo_enabled:
            nominal,nf,np,nn,ns=_qpsk_fit(row,symbol_rate_baud,block_config,False)
            fit_count=min(row.numel(),block_config.maximum_fit_samples)
            resolvable=max(1.,2*block_config.maximum_residual_cfo_hz/(symbol_rate_baud/(4*row.numel())))
            threshold=.5*math.log(fit_count)+math.log(resolvable)
            gain=float((s-ns)*fit_count)
            if gain<=threshold:
                recovered,f,p,n,s=nominal,nf,np,nn,ns
                nominal_used=True
        if not config.cpr_enabled:
            recovered=recovered*torch.exp(1j*p)
        rows.append(recovered);cfo.append(float(f));phase.append(float(p));noise.append(float(n));score.append(float(s))
        gates.append(nominal_used);gains.append(gain);thresholds.append(threshold)
    return torch.stack(rows),{"receiver_version":QPSK_BLOCK_DSP_VERSION,
        "estimated_cfo_hz":cfo,"estimated_phase_start_rad":phase,
        "estimated_normalized_noise_variance":noise,"marginal_log_likelihood":score,
        "nominal_cfo_prior_used":gates,"frequency_loglikelihood_gain":gains,
        "frequency_evidence_threshold":thresholds,
        "maximum_residual_cfo_hz":block_config.maximum_residual_cfo_hz,
        "phase_symmetry_unwrap_events":[0]*received.shape[0],
        "experimental_candidate":True,"failure_policy":"estimates/errors retained, no frame rejection"}
