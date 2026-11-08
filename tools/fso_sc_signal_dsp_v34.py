"""Short-frame 16QAM carrier-phase recovery without local symmetry unwrapping.

Only CPR changes.  The archived fourth-power CFO estimator and positive real
block-amplitude side information are retained exactly.  One bounded affine
phase trajectory is fitted by the received-sample constellation-marginal AWGN
likelihood, instead of unwrapping independent 65-symbol blind-phase windows.

The frame-wide slow-phase model is motivated by the present short frames and
1-kHz LO linewidth.  It is not the authors' undisclosed DSP and can fail on
long frames, fast phase wander, strongly biased constellations, or false CFO
acquisition.  The uniform-constellation likelihood and received-energy noise
estimate are assumptions.  No transmitted symbol, true channel phase/CFO,
codec feedback, parity check or image-quality metric enters this receiver.

The fundamental pi/2 ambiguity remains: initial phase is represented in the
branch nearest zero, as in the original simulator.  No pilots are added.
"""
from dataclasses import asdict, dataclass
import math

import torch

if 'recover_qam_carrier_v3' not in globals():
    from fso_sc_signal_dsp_v3 import recover_qam_carrier_v3
if '_estimate_cfo_fourth_power' not in globals():
    from fso_sc_signal_dsp import SignalDSPConfig, _estimate_cfo_fourth_power


QAM16_BLOCK_DSP_VERSION_V34 = '16qam_same_coarse_cfo_block_affine_marginal_ml_cpr_v34_1'


@dataclass(frozen=True)
class QAM16BlockCPRConfigV34:
    phase_trials: int = 64
    span_trials: int = 9
    refinement_steps: int = 6
    maximum_fit_samples: int = 2048
    candidate_chunk_size: int = 32
    minimum_estimated_noise_variance: float = 1e-3
    # One *unoversampled* fourth-power carrier bin spans pi/2 phase over N.
    maximum_phase_span_rad: float = math.pi / 2

    def __post_init__(self):
        for name in ('phase_trials','span_trials','refinement_steps',
                     'maximum_fit_samples','candidate_chunk_size'):
            if type(getattr(self, name)) is not int:
                raise ValueError('Integer block-CPR resolution required: '+name)
        if (self.phase_trials < 8 or self.span_trials < 3 or self.span_trials % 2 != 1
                or self.refinement_steps < 0 or self.maximum_fit_samples < 8
                or self.candidate_chunk_size < 1):
            raise ValueError('Invalid block-CPR grid or refinement budget')
        for name in ('minimum_estimated_noise_variance','maximum_phase_span_rad'):
            value = getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite block-CPR parameter required: '+name)
        if self.maximum_phase_span_rad > math.pi/2:
            raise ValueError('Residual span must stay within one unoversampled fourth-power carrier bin')


def qam16_block_cpr_metadata_v34(config=None):
    config = QAM16BlockCPRConfigV34() if config is None else config
    if not isinstance(config,QAM16BlockCPRConfigV34):
        raise TypeError('Supply QAM16BlockCPRConfigV34')
    return dict(receiver_version=QAM16_BLOCK_DSP_VERSION_V34, affected_modulation='16QAM',
        config=asdict(config), coarse_CFO='Unchanged archived fourth-power FFT/Newton estimator',
        phase_model='One affine trajectory, parameterized at frame center to decouple phase and residual slope; no local phase unwrap',
        likelihood='Uniform square-16QAM symbol-marginal complex-AWGN likelihood, factorized into two 4-PAM axes',
        noise_estimate='max(mean normalized received symbol energy - 1, configured positive floor); no true noise PSD',
        initial_phase_ambiguity='pi/2 ambiguity represented by the initial branch nearest zero; not pilot-resolved',
        short_frame_assumption=True, true_phase_or_CFO_used=False, transmitted_bits_or_symbols_used=False,
        lpips_or_decoder_feedback_used=False, pilot_symbols_added=0, automatically_replaces_primary=False,
        failure_policy='All frames, estimates and errors retained; no data-dependent rejection',
        scope='Independent CPR engineering candidate, not author VPI recovery or numerical paper alignment',
        related_work=['https://doi.org/10.1109/JLT.2008.2010511',
                      'https://opg.optica.org/jlt/abstract.cfm?uri=jlt-31-13-2055'])


def _qam16_phase_branch_v34(value):
    return torch.remainder(value+math.pi/4,math.pi/2)-math.pi/4


def _qam16_log_likelihood_v34(sample,time,phase,span,noise,config):
    """Score candidate paths in bounded chunks; exp/log remain stable at high SNR."""
    levels = torch.tensor([-3.,-1.,1.,3.],device=sample.device,dtype=sample.real.dtype)/math.sqrt(10.)
    scores=[]
    for first in range(0,phase.numel(),config.candidate_chunk_size):
        last=first+config.candidate_chunk_size
        trajectory=phase[first:last,None]+span[first:last,None]*time
        rotated=sample[None,:]*torch.exp(-1j*trajectory)
        xi=-(rotated.real[...,None]-levels).square()/noise
        xq=-(rotated.imag[...,None]-levels).square()/noise
        scores.append((torch.logsumexp(xi,-1)+torch.logsumexp(xq,-1)).mean(-1))
    return torch.cat(scores)


def _qam16_choose_v34(scores,phase,span):
    # Only floating-point ties, not a low-SNR/acquisition success gate.  The
    # near-zero branch convention is explicit and does not use source truth.
    tolerance=8*torch.finfo(scores.dtype).eps*(1+scores.abs().amax())
    eligible=scores>=scores.amax()-tolerance
    order=_qam16_phase_branch_v34(phase-span/2).abs()+torch.finfo(scores.dtype).eps*span.abs()
    return torch.where(eligible,order,torch.full_like(order,float('inf'))).argmin()


def _qam16_fit_affine_phase_v34(observed,config):
    count=observed.shape[-1];dtype=observed.real.dtype;device=observed.device
    ids=torch.linspace(0,count-1,min(count,config.maximum_fit_samples),device=device).round().long()
    # Centering prevents slope-grid refinement from moving the intercept at
    # the midpoint. Otherwise a coarse slope can trap the finer phase search.
    time=ids.to(dtype)/(count-1)-.5
    phase_grid=(torch.arange(config.phase_trials,device=device,dtype=dtype)
                /config.phase_trials-.5)*(math.pi/2)
    span_grid=torch.linspace(-config.maximum_phase_span_rad,config.maximum_phase_span_rad,
                             config.span_trials,device=device,dtype=dtype)
    phase_seeds=phase_grid.repeat_interleave(config.span_trials)
    span_seeds=span_grid.repeat(config.phase_trials)
    start_phases=[];spans=[];noise_estimates=[];fit_scores=[];bound_hits=[]
    offsets=torch.tensor([-1.,0.,1.],device=device,dtype=dtype)
    for row in observed:
        noise=(row.abs().square().mean()-1.).clamp_min(config.minimum_estimated_noise_variance)
        sample=row[ids]
        phase=phase_seeds.clone();span=span_seeds.clone()
        scores=_qam16_log_likelihood_v34(sample,time,phase,span,noise,config)
        chosen=_qam16_choose_v34(scores,phase,span)
        best_phase=phase[chosen];best_span=span[chosen]
        phase_step=math.pi/(2*config.phase_trials)
        span_step=2*config.maximum_phase_span_rad/(config.span_trials-1)
        for _ in range(config.refinement_steps):
            phase_step/=2;span_step/=2
            phase=_qam16_phase_branch_v34(best_phase+offsets.repeat_interleave(3)*phase_step)
            span=(best_span+offsets.repeat(3)*span_step).clamp(
                -config.maximum_phase_span_rad,config.maximum_phase_span_rad)
            scores=_qam16_log_likelihood_v34(sample,time,phase,span,noise,config)
            chosen=_qam16_choose_v34(scores,phase,span)
            best_phase=phase[chosen];best_span=span[chosen]
        # Score the selected trajectory rather than an arbitrary last-grid entry.
        final_score=_qam16_log_likelihood_v34(sample,time,best_phase.reshape(1),best_span.reshape(1),noise,config)[0]
        start_phases.append(_qam16_phase_branch_v34(best_phase-best_span/2))
        spans.append(best_span);noise_estimates.append(noise);fit_scores.append(final_score)
        bound_hits.append(bool(best_span.abs()>=config.maximum_phase_span_rad-2*torch.finfo(dtype).eps))
    initial=torch.stack(start_phases);span=torch.stack(spans)
    trajectory=initial[:,None]+span[:,None]*torch.arange(count,device=device,dtype=dtype)[None,:]/(count-1)
    return observed*torch.exp(-1j*trajectory),trajectory,torch.stack(noise_estimates),torch.stack(fit_scores),bound_hits


@torch.no_grad()
def recover_qam_carrier_v34(received,modulation,symbol_rate_baud,config=None,
                           known_block_magnitude=None,*,block_config=None):
    """V3-compatible receiver: only 16QAM CPR changes; all other paths dispatch."""
    if modulation.upper()!='16QAM':
        return recover_qam_carrier_v3(received,modulation,symbol_rate_baud,config,known_block_magnitude)
    config=SignalDSPConfig() if config is None else config
    block_config=QAM16BlockCPRConfigV34() if block_config is None else block_config
    if not isinstance(block_config,QAM16BlockCPRConfigV34):
        raise TypeError('Supply QAM16BlockCPRConfigV34')
    if (not isinstance(received,torch.Tensor) or received.ndim!=2 or not torch.is_complex(received)
            or received.shape[0]<1 or received.shape[-1]<8 or not bool(torch.isfinite(received).all())):
        raise ValueError('Expected finite complex [batch,symbols>=8] observations')
    if type(symbol_rate_baud) not in (int,float) or not math.isfinite(symbol_rate_baud) or symbol_rate_baud<=0:
        raise ValueError('Positive finite symbol rate required')
    if config.amplitude_mode!='known_block_magnitude':
        raise ValueError('Retain the declared positive real block-amplitude side information')
    if (not isinstance(known_block_magnitude,torch.Tensor) or torch.is_complex(known_block_magnitude)
            or not bool(torch.isfinite(known_block_magnitude).all()) or not bool((known_block_magnitude>0).all())):
        raise ValueError('Positive finite real block magnitude required, not true complex gain')
    amplitude=torch.broadcast_to(known_block_magnitude,(received.shape[0],1))
    observed=received/amplitude
    zeros=torch.zeros(received.shape[0],device=received.device,dtype=received.real.dtype)
    if config.cfo_enabled:
        observed,cfo_hz,peak_ratio=_estimate_cfo_fourth_power(observed,symbol_rate_baud,config)
    else:
        cfo_hz,peak_ratio=zeros,zeros
    if config.cpr_enabled:
        observed,phase,noise,scores,bound_hits=_qam16_fit_affine_phase_v34(observed,block_config)
    else:
        phase=torch.zeros_like(received.real);noise=zeros;scores=zeros;bound_hits=[False]*received.shape[0]
    if not bool(torch.isfinite(observed).all()):
        raise FloatingPointError('Nonfinite block-CPR output')
    diagnostics=dict(receiver_version=QAM16_BLOCK_DSP_VERSION_V34,
        amplitude_mode=config.amplitude_mode, estimated_cfo_hz=cfo_hz.detach().cpu().tolist(),
        coarse_cfo_hz=cfo_hz.detach().cpu().tolist(), coarse_cfo_estimator_unchanged=True,
        cfr_peak_to_mean_power=peak_ratio.detach().cpu().tolist(),
        cfr_unambiguous_interval_hz=[-symbol_rate_baud/8,symbol_rate_baud/8],
        estimated_phase_start_rad=phase[:,0].detach().cpu().tolist(),
        estimated_phase_end_rad=phase[:,-1].detach().cpu().tolist(),
        estimated_phase_span_rad=(phase[:,-1]-phase[:,0]).detach().cpu().tolist(),
        maximum_allowed_phase_span_rad=block_config.maximum_phase_span_rad,
        estimated_normalized_noise_variance=noise.detach().cpu().tolist(),
        marginal_log_likelihood=scores.detach().cpu().tolist(), phase_span_bound_hit=bound_hits,
        phase_symmetry_unwrap_events=[0]*received.shape[0], local_phase_unwrap_used=False,
        single_affine_phase_trajectory=True, cpr_enabled=config.cpr_enabled,
        residual_phase_slope_is_coarse_cfo_change=False,
        failure_policy='All estimated frames and errors retained; no frame rejection',
        uses_true_phase_or_frequency=False, uses_transmitted_bits_or_symbols=False)
    return observed,diagnostics
