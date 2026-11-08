# Run on your own inputs or replay a historical experiment

[English README](../README.md) · [한국어 README](../README.ko.md) · [Latest results](LATEST_RESULTS.md)

**Nothing needs to be copied from Drive to publish this repository.** The release includes source and aggregate results, not the underlying images, checkpoints, original run manifests, native BPG bitstreams, or private per-image outputs.

## Run on your own data

Use [01](../notebooks/01_prepare_GEE.ipynb), [02](../notebooks/02_train.ipynb), and [03](../notebooks/03_final_holdout.ipynb) in that order. Provide your own authorized inputs under `MyDrive/FSO_data/`:

| Input | Count and role |
| --- | --- |
| `imagenet_train_10000.zip` | 10,000 training images. No full ImageNet TAR in Drive. |
| `GEE_train_fixed.zip` | 1,000 remote-sensing training images. This reproduction used a NAIP proxy. |
| `GEE_test.zip` | 1,000 candidate evaluation images, separate from training. |

See [data preparation](DATA.md). Use the same `RUN_NAME` for 02 and 03. This workflow implements the **earlier baseline**, not the exact v25 training run. Do not expect the published v25 scores from a new model, even if the counts match.

## Replay the locked v25 experiments

These are records of executed experiments, not a quick start for a new dataset. All five locked evaluation notebooks need the **original, unchanged** files below. Restore them to your own Drive from a private backup if you already own them:

```text
MyDrive/FSO_data/
├── GEE_test.zip
└── stage2_error_range_to1_v25/
    └── extension_cc8ff5b866b4c901/
        ├── stage2_error_range_to1_summary.json
        ├── configuration.json
        └── resume/
            ├── stage2_resume_a.pth
            ├── stage2_resume.json       # also checked by v27 / v31
            └── stage2_store.json        # also checked by v27 / v31
```

The `extension_…` directory is located by the original summary SHA, not by taking the newest file. The v25 checkpoint is **3,250,750,999 bytes**, SHA-256 `c4443bd14c0e3292e4a456a715e5de27a42855097a46244d93fc573fb656f121`. The notebooks use its fixed weights for inference and do not restore the optimizer. Keeping the whole original `resume/` directory privately is safer than reconstructing individual metadata files.

| Replay | Additional private input |
| --- | --- |
| [v27](../notebooks/frozen/FSO_SC_paper_figures_full_v27.ipynb) | None beyond the v25 bundle above. The official BPG codec is downloaded/built in Colab. |
| [v31](../notebooks/frozen/FSO_SC_power_extension_v31.ipynb) | Original `stage2_paper_sweep_v27/frozen_882a885033790ab2/`: `paper_sweep_summary.json`, `sources/selected_native_bpg.zip`, and the 24 shared `cases/<id>/metrics.json` files. Preserve the whole v27 run directory privately. |
| [v32](../notebooks/frozen/FSO_SC_stage_effect_v32.ipynb) | `rebuild_v12/rebuild_v12_run1/stage1_final.pth`, the original completed Stage 1, not an early epoch. |
| [v33](../notebooks/frozen/FSO_SC_receiver_transition_v33.ipynb) / [v34](../notebooks/frozen/FSO_SC_qam16_phase_repair_v34.ipynb) | None. No Stage 1 or v27 output is read. |

The locked inference notebooks do not need the ImageNet or NAIP training ZIPs. The small published `*_aggregate.json` files **cannot** stand in for the original pinned summaries. The notebooks verify input hashes, but small numerical differences can still occur with a different hardware or runtime. See each notebook for storage and compute requirements.

## How the frozen model was produced

The v25 model used in the latest figures was initialized from the completed [v12 clean Stage 1](../notebooks/training_history/FSO_SC_rebuild_v12.ipynb) (100 epochs), followed by:

1. [v21 uniform-wrong continuation](../notebooks/training_history/FSO_SC_stage2_controlled_error_v21.ipynb): 512 optimizer updates; controlled counts of 10, 51, 102, or 154 wrong indices per 1,024-token image.
2. [v24 wide-range continuation](../notebooks/training_history/FSO_SC_stage2_error_range_control_v24.ipynb): 512 more updates, switching to 10, 102, 307, or 512 wrong indices.
3. [v25 wide1375](../notebooks/training_history/FSO_SC_stage2_error_range_to1_v25.ipynb): 351 more updates with that policy; 1,375 cumulative Stage 2 updates, one full pass worth of image presentations.

These are independent corruption policies, not recovered author settings or a claim that the errors have the exact physical-channel distribution. The archived notebooks document the code and provenance. They refer to the original data, checkpoints, and earlier diagnostic summaries (including the v23 error-support report for v24) and are **not** a self-contained replacement for the 01–03 fresh-data workflow. The original v12 notebook also contains a separate Stage 2 experiment; that run is not the ancestor of v25.

## Maintainer checklist

- Keep the original inputs and full run folders privately on Drive/NAS if you want exact historical replays.
- Publish only this source-only directory. Do not commit model files, original data, per-image archives, credentials, or private download links.
- If you later choose to distribute a trained checkpoint or data, make that a separate provenance and rights decision. Do not imply that the existing release already includes them.

한국어 요약: **GitHub 공개를 위해 Drive 파일을 옮길 필요는 없습니다.** 위 경로는 본인이 당시 실험을 다시 실행할 때만 필요합니다. 공개 저장소에서 새 데이터로 실행하려면 01–03번을 사용하고, v25의 고정 결과는 집계·그래프로 확인하세요.
