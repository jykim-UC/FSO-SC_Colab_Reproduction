"""BPG/real-soft-LDPC comparison over the unchanged symbol-rate FSO/DSP.

No codec substitute, BER threshold erasure, truth-aided decoder or training.
Known file length, real block-amplitude CSI and noise PSD are disclosed receiver
assumptions. The simulator's phase and transmitted bits are not passed to DSP
or LDPC. Transmitted bits are used only for AFTER-decoding error statistics.
"""
import hashlib
import json
import math

import numpy as np
import torch


BPG_EVALUATION_VERSION = "fso_sc_bpg_comparison_v1"


def bpg_comparison_conditions():
    """14 predeclared development conditions, not the entire paper sweep."""
    conditions = []
    def add(group,power,modulation,zenith,pointing=20.,weather="clear"):
        kwargs = {"launch_power_dbm":float(power),"modulation":modulation,
                  "zenith_angle_deg":float(zenith),"pointing_error_urad":float(pointing),
                  "background_light_radiance_w_m2_sr_um":1e-4,
                  "weather":weather,"scattered_power_ratio":.95}
        key = json.dumps(kwargs,sort_keys=True,separators=(",",":"))
        conditions.append({"id":hashlib.sha256(key.encode()).hexdigest()[:16],
                           "group":group,"channel_kwargs":kwargs})
    for power in (30,33,36):
        for modulation in ("QPSK","8QAM","16QAM"):
            add("power_at_zenith30",power,modulation,30)
    for modulation in ("QPSK","8QAM","16QAM"):
        add("coverage_at_36dBm",36,modulation,60)
    for weather in ("snow","fog"):
        add("adverse_weather",36,"QPSK",0,pointing=15,weather=weather)
    return conditions


def soft_qam_llr(symbols,modulation,noise_variance,namespace,clip=30.):
    """Exact log-MAP bit LLR, log(P0/P1), E|complex noise|²=N0.

    The equalized noise variance uses disclosed real-amplitude/noise-PSD CSI,
    not the simulator's actual noise sample or exact phase trajectory. Residual
    phase-estimation errors are deliberately NOT oracle-corrected here.
    """
    if not torch.is_complex(symbols) or symbols.ndim != 2:
        raise ValueError("Expected complex [batch,symbols]")
    points,labels,_ = namespace["qam_constellation"](modulation,symbols.device)
    n0 = torch.as_tensor(noise_variance,device=symbols.device,dtype=torch.float64)
    if not bool(torch.isfinite(n0).all()) or bool((n0 < 0).any()):
        raise ValueError("Noise variance must be finite and nonnegative")
    if n0.ndim == 0:
        n0 = n0.reshape(1,1)
    n0 = torch.broadcast_to(n0,symbols.shape).clamp_min(1e-15)
    # Float64 avoids overflow/cancellation in the severe-fog diagnostic. No
    # precision change is made to the semantic model or the carrier recovery.
    distances = (symbols.to(torch.complex128)[...,None]-points.to(torch.complex128)).abs().square()
    scores = -distances/n0[...,None]
    llr = torch.stack([torch.logsumexp(scores[...,labels[:,bit]==0],dim=-1)
                       -torch.logsumexp(scores[...,labels[:,bit]==1],dim=-1)
                       for bit in range(labels.shape[1])],dim=-1)
    if not bool(torch.isfinite(llr).all()):
        raise FloatingPointError("Nonfinite soft QAM LLR; do not drop the frame")
    return llr.clamp(-clip,clip).to(torch.float32)


@torch.inference_mode()
def transmit_bpg_bytes(payload,physical,condition,namespace,seed,device="cpu",ldpc=None,dsp_config=None):
    """Actually transmit every BPG byte; optional true LDPC, then estimated DSP.

    File length is known framing side information (not a protected header or
    CRC). QAM tail padding and complete LDPC block padding are counted.
    """
    if not isinstance(payload,bytes) or not payload:
        raise ValueError("A nonempty actual BPG byte stream is required")
    device = torch.device(device)
    source = torch.from_numpy(np.unpackbits(np.frombuffer(payload,dtype=np.uint8)).copy()).to(device)
    source_count = source.numel()
    if ldpc is None:
        transmitted_bits = source.reshape(1,-1)
        blocks,information_padding = 0,0
    else:
        blocks = math.ceil(source_count/ldpc.k)
        information_padding = blocks*ldpc.k-source_count
        padded = torch.cat((source,torch.zeros(information_padding,dtype=torch.uint8,device=device)))
        transmitted_bits = ldpc.encode_bits(padded.reshape(blocks,ldpc.k)).reshape(1,-1)
    modulation = condition["channel_kwargs"]["modulation"]
    _,_,bits_per_symbol = namespace["qam_constellation"](modulation,device)
    coded_count = transmitted_bits.numel()
    qam_padding = (-coded_count)%bits_per_symbol
    bits_for_qam = torch.cat((transmitted_bits,torch.zeros((1,qam_padding),device=device,dtype=torch.uint8)),dim=-1)
    symbols = namespace["qam_modulate"](bits_for_qam.reshape(1,-1,bits_per_symbol),modulation)
    kwargs = {key:value for key,value in condition["channel_kwargs"].items() if key != "modulation"}
    observation = physical(symbols,generator=torch.Generator(device=device).manual_seed(int(seed)),**kwargs)
    if observation.is_equalized or observation.channel_gain is None:
        raise ValueError("Need an unequalized single-Rx physical observation")
    config = namespace["SignalDSPConfig"]() if dsp_config is None else dsp_config
    if config.amplitude_mode != "known_block_magnitude":
        raise ValueError("This baseline declares real block-amplitude CSI; no hidden alternate estimator")
    magnitude = observation.channel_gain.abs().mean(dim=-1,keepdim=True)
    recovered,dsp_diagnostics = namespace["recover_qam_carrier"](
        observation.symbols,modulation,physical.config.symbol_rate_baud,config,
        known_block_magnitude=magnitude)
    hard = namespace["qam_demodulate"](recovered,modulation).reshape(1,-1)[:,:coded_count]
    raw_errors = int((hard != transmitted_bits).sum())
    budget = observation.diagnostics["budget"]
    receiver_noise = (budget["background_noise_w"] if physical.config.include_background_noise else 0.)
    receiver_noise += budget["receiver_shot_noise_w"] if physical.config.include_shot_noise else 0.
    equivalent_n0 = receiver_noise/magnitude.to(torch.float64).square().clamp_min(1e-300)
    if physical.config.include_ase_noise:
        equivalent_n0 += budget["ase_noise_w"]/budget["launch_power_w"]
    ldpc_statistics = None
    if ldpc is None:
        decoded_bits = hard.reshape(-1)[:source_count]
    else:
        soft = soft_qam_llr(recovered,modulation,equivalent_n0,namespace,clip=ldpc.llr_clip)
        llr = soft.reshape(-1)[:coded_count].reshape(blocks,ldpc.n)
        information,ldpc_statistics = ldpc.decode_llr(llr)
        decoded_bits = information.reshape(-1)[:source_count]
    received_bytes = np.packbits(decoded_bits.cpu().numpy()).tobytes()
    result = {"received_bytes":received_bytes,"seed":int(seed),
              "bit_errors":int((decoded_bits != source).sum()),"valid_bits":source_count,
              "pre_fec_bit_errors":raw_errors,"pre_fec_valid_bits":coded_count,
              "source_bytes":len(payload),"source_bits":source_count,
              "coded_bits":coded_count,"qam_padding_bits":qam_padding,
              "transmitted_bits":coded_count+qam_padding,"channel_symbols":symbols.numel(),
              "ldpc_blocks":blocks,"information_padding_bits":information_padding,
              "noise_variance_equalized":float(equivalent_n0.mean()),
              "weak_CFR_peak_frames_diagnostic_only":sum(v<10 for v in dsp_diagnostics["cfr_peak_to_mean_power"]),
              "ldpc_nonconverged_blocks":0,"ldpc_iterations_mean":0.,
              "frame_integrity_policy":"Declared BPG header validation/native decoding decides usability; no CRC or truth-based erasure"}
    if ldpc_statistics is not None:
        result.update(ldpc_nonconverged_blocks=int((~ldpc_statistics["converged"]).sum()),
                      ldpc_iterations_mean=float(ldpc_statistics["iterations"].float().mean()))
    return result


@torch.inference_mode()
def evaluate_bpg_condition(codec,streams,images_cpu,metric,physical,condition,names,namespace,
                           device="cpu",ldpc=None,base_seed=20261028,progress=None):
    """Use Fig.4's explicit failure marker plus a separately measured gray LPIPS.

    A successful (possibly damaged/mosaic) BPG image is scored as decoded. Bit
    errors or failed LDPC parity do not force an erasure. Decoder/library errors
    that indicate missing software must propagate rather than become failures.
    """
    if len(images_cpu) != len(names) or set(names) != set(streams) or len(set(names)) != len(names):
        raise ValueError("Image/name/actual BPG stream coverage differs")
    device = torch.device(device)
    records = []
    for index,name in enumerate(names):
        source = streams[name]
        seed = namespace["channel_seed"](condition,name,0,base_seed)
        if physical is None:
            result = {"received_bytes":source,"seed":seed,"bit_errors":0,"valid_bits":len(source)*8,
                      "pre_fec_bit_errors":0,"pre_fec_valid_bits":0,"source_bytes":len(source),
                      "source_bits":len(source)*8,"coded_bits":0,"qam_padding_bits":0,
                      "transmitted_bits":0,"channel_symbols":0,"ldpc_blocks":0,
                      "information_padding_bits":0,"ldpc_nonconverged_blocks":0,"ldpc_iterations_mean":0.}
        else:
            result = transmit_bpg_bytes(source,physical,condition,namespace,seed,device,ldpc)
        try:
            decoded = codec.decode(result.pop("received_bytes"))
        except Exception as error:
            if hasattr(error,"add_note"):
                error.add_note(f"BPG evaluation image={name}, condition={condition.get('id')}")
            raise
        failed = not decoded.ok
        height,width = images_cpu.shape[-2:]
        if failed:
            raster = np.full((height,width,3),128,dtype=np.uint8)
        else:
            raster = np.asarray(decoded.image,dtype=np.uint8)
            if raster.shape != (height,width,3):
                raise ValueError("Codec contract violated: successful decode with wrong shape")
        prediction = torch.from_numpy(raster.copy()).permute(2,0,1).unsqueeze(0).to(device,dtype=torch.float32)/127.5-1.
        target = images_cpu[index:index+1].to(device)
        actual = metric(prediction,target).reshape(-1)
        if actual.numel() != 1 or not bool(torch.isfinite(actual).all()):
            raise FloatingPointError(f"Nonfinite LPIPS for image={name}; do not discard or replace the frame")
        measured = float(actual[0])
        record = {"image_name":str(name),"repeat":0,"lpips":1. if failed else measured,
                  "rendered_lpips":measured,"decode_failed":failed,"decode_status":decoded.status,
                  "decode_error":decoded.error if failed else None,
                  "mse":float((prediction-target).square().mean()),
                  "lpips_policy":"1 is paper-style header-validation/native-decode failure marker; rendered_lpips is separately measured",
                  **result}
        records.append(record)
        if progress is not None:
            progress(index+1,len(names))
    summary = namespace["summarize_records"](records,seed=base_seed)
    summary.update(decode_failures=sum(r["decode_failed"] for r in records),
                   decode_failure_rate=float(np.mean([r["decode_failed"] for r in records])),
                   rendered_lpips_mean=float(np.mean([r["rendered_lpips"] for r in records])),
                   source_bpp_mean=float(np.mean([r["source_bits"]/(height*width) for r in records])),
                   transmitted_bpp_mean=float(np.mean([r["transmitted_bits"]/(height*width) for r in records])),
                   channel_symbols_mean=float(np.mean([r["channel_symbols"] for r in records])),
                   ldpc_nonconverged_blocks=sum(r["ldpc_nonconverged_blocks"] for r in records),
                   pre_fec_ber=sum(r["pre_fec_bit_errors"] for r in records)/max(sum(r["pre_fec_valid_bits"] for r in records),1),
                   failure_policy="Retain all images; declared BPG header-validation/native-decode failure -> paper marker1, companion measured gray LPIPS",
                   metric_scope="Source file bits only for post-FEC BER; coded bits including LDPC padding for pre-FEC BER")
    # BPG has no semantic indices. These denominators are absent rather than
    # evidence of an error-free index/symbol receiver.
    summary["index_error_rate"] = None
    summary["symbol_error_rate"] = None
    return {"version":BPG_EVALUATION_VERSION,"condition":condition,"records":records,"summary":summary}
