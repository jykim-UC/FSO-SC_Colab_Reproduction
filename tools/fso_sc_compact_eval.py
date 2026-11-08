"""Compact, paired FSO-SC evaluation helpers; no training or work at import.

The reduced grid, 200-image subset, seeds and bootstrap are reproduction choices,
not settings disclosed by the paper. All test results, including bad DSP frames,
are retained. The caller owns checkpoint loading, output files and resume logic.
"""
import hashlib
import json
import math

import numpy as np
import torch


COMPACT_EVAL_VERSION = "fso_sc_compact_eval_v1"
COMPACT_BASE_SEED = 20261028


def condition_key(condition):
    """Identity depends on physical parameters, never model/stage/group labels."""
    return json.dumps(condition["channel_kwargs"], sort_keys=True, separators=(",", ":"))


def compact_conditions():
    catalog = {}
    def add(group, power=36, modulation="QPSK", zenith=0, pointing=20,
            radiance=1e-4, weather="clear", rho=.95):
        kwargs = {"launch_power_dbm": float(power), "modulation": modulation,
                  "zenith_angle_deg": float(zenith), "pointing_error_urad": float(pointing),
                  "background_light_radiance_w_m2_sr_um": float(radiance),
                  "weather": weather, "scattered_power_ratio": float(rho)}
        key = json.dumps(kwargs, sort_keys=True, separators=(",", ":"))
        if key not in catalog:
            catalog[key] = {"id": hashlib.sha256(key.encode()).hexdigest()[:16],
                            "group": group, "groups": [group], "channel_kwargs": kwargs,
                            "label": (f"{modulation}; Pt={power:g}dBm; zenith={zenith:g}; "
                                      f"pointing={pointing:g}urad; BGL={radiance:g}; "
                                      f"weather={weather}; rho={rho:g}")}
        elif group not in catalog[key]["groups"]:
            catalog[key]["groups"].append(group)
        return catalog[key]
    main = [add("main", power=p, modulation=m, zenith=z)
            for p in (30, 33, 36) for m in ("QPSK", "8QAM", "16QAM") for z in (0, 30, 60, 80)]
    pointing = [add("pointing", modulation="16QAM", zenith=60, pointing=p) for p in (18, 20, 22)]
    background = [add("background", power=33, modulation="16QAM", pointing=19, radiance=b)
                  for b in (1e-4, 1e-2, 1)]
    weather = [add("weather", pointing=15, weather=w) for w in ("clear", "dust", "rain", "snow", "fog")]
    turbulence = [add("turbulence", power=33, modulation="16QAM", pointing=19, rho=r)
                  for r in (.95, .75, .596)]
    main_ids = {c["id"] for c in main}
    robustness = [c for c in catalog.values() if c["id"] not in main_ids]
    ablation = [c for c in main if c["channel_kwargs"]["launch_power_dbm"] == 33
                and c["channel_kwargs"]["zenith_angle_deg"] in (30, 60)]
    anchor_values = {(36, "QPSK", 0), (33, "8QAM", 60), (30, "16QAM", 80)}
    anchors = [c for c in main if (c["channel_kwargs"]["launch_power_dbm"],
                                  c["channel_kwargs"]["modulation"],
                                  c["channel_kwargs"]["zenith_angle_deg"]) in anchor_values]
    return {"main": main, "robustness": robustness, "all": list(catalog.values()),
            "pointing": pointing, "background": background, "weather": weather,
            "turbulence": turbulence, "ablation": ablation, "anchors": anchors}


def evaluation_plan(image_count=200):
    conditions = compact_conditions()
    anchor_ids = {c["id"] for c in conditions["anchors"]}
    stage2 = [{"condition": c, "repeats": 3 if c["id"] in anchor_ids else 1}
              for c in conditions["all"]]
    stage1 = [{"condition": c, "repeats": 1} for c in conditions["ablation"]]
    n = int(image_count)
    if n < 1:
        raise ValueError("image_count must be positive")
    counts = {"stage2_physical_decodes": sum(j["repeats"] for j in stage2) * n,
              "stage1_physical_decodes": len(stage1) * n, "clean_decodes": 2 * n}
    counts["total_decodes"] = sum(counts.values())
    return {"version": COMPACT_EVAL_VERSION, "image_count": n, "stage2": stage2,
            "stage1_ablation": stage1, "counts": counts, "conditions": conditions}


def select_test_names(names, count=200, seed=COMPACT_BASE_SEED, expected_total=1000):
    names = sorted(str(name) for name in names)
    if len(names) != len(set(names)):
        raise ValueError("Test image identifiers must be unique, including subdirectories")
    if expected_total is not None and len(names) != expected_total:
        raise ValueError(f"Expected {expected_total} test images, found {len(names)}")
    if not 1 <= count <= len(names):
        raise ValueError("Invalid test subset count")
    indices = np.random.default_rng(seed).choice(len(names), size=count, replace=False)
    return sorted(names[int(i)] for i in indices)


def channel_seed(condition, image_name, repeat, base_seed=COMPACT_BASE_SEED):
    material = json.dumps([int(base_seed), condition_key(condition), str(image_name), int(repeat)],
                          separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") % (2**63 - 1)


def payload_bpp(indices, image_hw, bits=14):
    if indices.ndim < 2 or indices.shape[0] < 1 or len(image_hw) != 2:
        raise ValueError("Indices must retain a batch dimension and image_hw must be (H,W)")
    h, w = map(int, image_hw)
    if min(h, w, bits) < 1:
        raise ValueError("Image dimensions and bit width must be positive")
    return indices.numel() * int(bits) / (indices.shape[0] * h * w)


def image_means(records, metric="lpips"):
    grouped = {}
    seen = set()
    for record in records:
        identifier = (record["image_name"], int(record["repeat"]))
        if identifier in seen:
            raise ValueError("Duplicate image/repeat record")
        seen.add(identifier)
        score = float(record[metric])
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite evaluation score")
        grouped.setdefault(identifier[0], []).append(score)
    if not grouped:
        raise ValueError("Cannot summarize zero images")
    return {name: float(np.mean(values)) for name, values in sorted(grouped.items())}


def bootstrap_mean_ci(values, seed=COMPACT_BASE_SEED, bootstrap_samples=2000):
    values = np.asarray(list(values), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or bootstrap_samples < 1:
        raise ValueError("Finite nonempty image-level values and positive bootstrap count required")
    draw = np.random.default_rng(seed).integers(0, len(values), size=(bootstrap_samples, len(values)))
    lower, upper = np.quantile(values[draw].mean(axis=1), (.025, .975))
    return {"mean": float(values.mean()), "ci95": [float(lower), float(upper)],
            "independent_images": len(values), "bootstrap_samples": int(bootstrap_samples),
            "ci_unit": "image; channel repeats averaged within each image"}


def summarize_records(records, seed=COMPACT_BASE_SEED, bootstrap_samples=2000):
    averaged = image_means(records)
    result = bootstrap_mean_ci(averaged.values(), seed, bootstrap_samples)
    result.update({"lpips_mean": result["mean"], "unique_images": len(averaged),
                   "image_channel_realizations": len(records), "per_image_lpips": averaged})
    count_keys = ("bit_errors", "valid_bits", "index_errors", "indices", "symbol_errors", "valid_symbols")
    totals = {key: sum(int(r.get(key, 0)) for r in records) for key in count_keys}
    result.update({"counts": totals,
                   "payload_ber": totals["bit_errors"] / max(totals["valid_bits"], 1),
                   "index_error_rate": totals["index_errors"] / max(totals["indices"], 1),
                   "symbol_error_rate": totals["symbol_errors"] / max(totals["valid_symbols"], 1),
                   "mse_mean": float(np.mean(list(image_means(records, "mse").values()))),
                   "failure_policy": "all finite frames retained, including bad DSP recovery; nonfinite raises"})
    return result


def paired_model_difference(stage2_records, stage1_records, seed=COMPACT_BASE_SEED, bootstrap_samples=2000):
    # Restrict repeated anchors to repeat0 when comparing the one-repeat ablation.
    right = {(r["image_name"], r["repeat"]): r for r in stage1_records}
    left = {(r["image_name"], r["repeat"]): r for r in stage2_records
            if (r["image_name"], r["repeat"]) in right}
    if len(left) != len(right) or not left:
        raise ValueError("Paired comparison requires identical image/repeat coverage")
    differences = []
    for key in sorted(right):
        a, b = left[key], right[key]
        if a.get("seed") != b.get("seed"):
            raise ValueError("Paired comparison requires the same channel seed")
        differences.append({"image_name": key[0], "repeat": key[1], "lpips": float(a["lpips"])-float(b["lpips"])})
    result = bootstrap_mean_ci(image_means(differences).values(), seed, bootstrap_samples)
    result.update({"difference": "Stage2 minus Stage1 LPIPS; negative favors Stage2",
                   "paired_realizations": len(differences)})
    return result


@torch.inference_mode()
def evaluate_condition(model, images_cpu, cached_indices_cpu, metric, channel, condition, names,
                       repeats=1, device="cuda", base_seed=COMPACT_BASE_SEED,
                       progress=None, sample_callback=None, sample_names=()):
    """Decode cached indices only; no encoder calls, optimizer steps or file writes.

    Set channel=None for a clean, error-free semantic reconstruction. Physical
    channel receives a fresh local generator per image/repeat, shared by stages.
    """
    if repeats < 1 or len(images_cpu) != len(names) or len(cached_indices_cpu) != len(names):
        raise ValueError("Invalid repetitions or cache/image/name lengths")
    if not len(names) or len(set(names)) != len(names):
        raise ValueError("Require nonempty unique image identifiers")
    model.eval()
    if hasattr(metric, "eval"):
        metric.eval()
    device = torch.device(device)
    samples = set(sample_names)
    records = []
    for i, name in enumerate(names):
        images = images_cpu[i:i+1].to(device)
        indices = cached_indices_cpu[i:i+1].to(device=device, dtype=torch.long)
        for repeat in range(repeats):
            seed = channel_seed(condition, name, repeat, base_seed)
            if channel is None:
                received, statistics = indices, {"valid_bits": indices.numel()*14, "indices": indices.numel()}
            else:
                received, statistics = channel(indices, return_statistics=True,
                                               generator=torch.Generator(device=device).manual_seed(seed),
                                               **condition["channel_kwargs"])
            prediction = model.decode_indices(received, received.shape)
            if prediction.shape != images.shape or not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(f"Invalid reconstruction for {name}, repeat {repeat}; frame not discarded")
            clipped = prediction.clamp(-1, 1)
            value = metric(clipped, images).reshape(-1)
            if value.numel() != 1 or not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"Invalid LPIPS for {name}, repeat {repeat}")
            record = {"image_name": str(name), "repeat": repeat, "seed": seed,
                      "lpips": float(value[0]), "mse": float((clipped-images).square().mean()),
                      "mse_raw": float((prediction-images).square().mean()),
                      "clipped_fraction": float((prediction.abs()>1).float().mean())}
            for key in ("bit_errors", "valid_bits", "index_errors", "indices", "symbol_errors", "valid_symbols"):
                record[key] = int(statistics.get(key, 0))
            diagnostics = statistics.get("channel", {}).get("signal_dsp", {})
            record["weak_CFR_peak_frames_diagnostic_only"] = sum(
                float(strength) < 10 for strength in diagnostics.get("cfr_peak_to_mean_power", []))
            records.append(record)
            if sample_callback is not None and name in samples and repeat == 0:
                sample_callback(name, repeat, clipped.detach().cpu())
            if progress is not None:
                progress(len(records), len(names)*repeats)
    return {"version": COMPACT_EVAL_VERSION, "condition": condition, "repeats": int(repeats),
            "records": records, "summary": summarize_records(records, seed=base_seed)}
