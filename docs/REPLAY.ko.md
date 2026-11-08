# 자체 데이터로 실행하거나 기존 실험 다시 실행하기

[English](REPLAY.md) · [한국어 README](../README.ko.md) · [최신 결과](LATEST_RESULTS.ko.md)

**GitHub에 이 저장소를 공개할 때 Drive에서 추가로 가져올 파일은 없습니다.** 공개본에는 코드와 집계 결과만 포함됩니다. 원본 영상, 체크포인트, 원본 실행 매니페스트, BPG 비트스트림, 이미지별 비공개 출력은 포함하지 않습니다.

## 자체 데이터로 실행

[01 데이터 검사](../notebooks/01_prepare_GEE.ipynb) → [02 모델 학습](../notebooks/02_train.ipynb) → [03 평가](../notebooks/03_final_holdout.ipynb) 순서로 실행합니다. 사용 권한이 있는 입력을 `MyDrive/FSO_data/` 아래에 준비하세요.

| 입력 파일 | 수량과 용도 |
| --- | --- |
| `imagenet_train_10000.zip` | 학습 이미지 10,000장. 전체 ImageNet TAR를 Drive에 올릴 필요는 없습니다. |
| `GEE_train_fixed.zip` | 원격탐사 학습 이미지 1,000장. 이번 재현에서는 NAIP 대체 데이터를 사용했습니다. |
| `GEE_test.zip` | 학습용 데이터와 분리된 평가 후보 이미지 1,000장. |

[데이터 준비](DATA.md)를 참고하고, 02와 03에서 같은 `RUN_NAME`을 사용하세요. 이 절차는 **이전 기본 모델**을 학습하는 방법이며, v25 학습을 그대로 다시 실행하는 절차는 아닙니다. 데이터 수가 같아도 새 모델에서 공개된 v25 점수가 그대로 나오지는 않습니다.

<a name="replay-the-locked-v25-experiments"></a>

## 고정된 v25 실험 다시 실행

아래 노트북 5개는 새 데이터용 예제가 아니라 이미 수행한 실험을 다시 실행하기 위한 것입니다. **당시와 동일한 원본 파일**이 필요합니다. 개인 백업을 보유한 경우 아래 경로로 복원하세요.

```text
MyDrive/FSO_data/
├── GEE_test.zip
└── stage2_error_range_to1_v25/
    └── extension_cc8ff5b866b4c901/
        ├── stage2_error_range_to1_summary.json
        ├── configuration.json
        └── resume/
            ├── stage2_resume_a.pth
            ├── stage2_resume.json       # v27 / v31에서 확인
            └── stage2_store.json        # v27 / v31에서 확인
```

`extension_…` 디렉터리는 가장 최근 파일이 아니라 원본 요약 파일의 SHA로 찾습니다. v25 체크포인트 크기는 **3,250,750,999바이트**이고 SHA-256은 `c4443bd14c0e3292e4a456a715e5de27a42855097a46244d93fc573fb656f121`입니다. 노트북은 고정 가중치로 추론하며 옵티마이저 상태를 복원하지 않습니다. 개별 메타데이터를 다시 만들기보다 원본 `resume/` 전체를 비공개로 보관하는 편이 안전합니다.

| 실험 | 추가 비공개 입력 |
| --- | --- |
| [v27](../notebooks/frozen/FSO_SC_paper_figures_full_v27.ipynb) | 위 v25 묶음 외에 없음. 공식 BPG 코덱은 Colab에서 내려받아 빌드합니다. |
| [v31](../notebooks/frozen/FSO_SC_power_extension_v31.ipynb) | 원본 `stage2_paper_sweep_v27/frozen_882a885033790ab2/`의 `paper_sweep_summary.json`, `sources/selected_native_bpg.zip`, 재사용하는 `cases/<id>/metrics.json` 24개. v27 실행 폴더 전체를 비공개로 보관하세요. |
| [v32](../notebooks/frozen/FSO_SC_stage_effect_v32.ipynb) | `rebuild_v12/rebuild_v12_run1/stage1_final.pth`. 중간 에포크가 아닌 Stage 1 완료 체크포인트가 필요합니다. |
| [v33](../notebooks/frozen/FSO_SC_receiver_transition_v33.ipynb) / [v34](../notebooks/frozen/FSO_SC_qam16_phase_repair_v34.ipynb) | 추가 입력 없음. Stage 1 체크포인트나 v27 출력을 읽지 않습니다. |

고정 추론 노트북에는 ImageNet이나 NAIP의 **학습용** ZIP이 필요하지 않습니다. 공개된 `*_aggregate.json`은 당시 원본 요약 파일을 대신할 수 없습니다. 노트북은 입력 해시를 확인하지만, 하드웨어나 런타임이 바뀌면 작은 수치 차이는 생길 수 있습니다. 저장 공간과 연산 요구사항은 각 노트북을 참고하세요.

<a name="how-the-frozen-model-was-produced"></a>

## 고정 모델의 학습 이력

최신 결과에 사용한 v25 모델은 100에포크를 마친 [v12 무채널 Stage 1](../notebooks/training_history/FSO_SC_rebuild_v12.ipynb)에서 시작해 다음 순서로 추가 학습했습니다.

1. [v21 — 균등한 오인덱스 주입](../notebooks/training_history/FSO_SC_stage2_controlled_error_v21.ipynb): 옵티마이저 업데이트 512회. 이미지당 1,024개 토큰 중 10, 51, 102, 154개를 잘못된 인덱스로 바꿉니다.
2. [v24 — 오류 범위 확대](../notebooks/training_history/FSO_SC_stage2_error_range_control_v24.ipynb): 업데이트 512회 추가. 바꾸는 인덱스 수를 10, 102, 307, 512개로 바꿨습니다.
3. [v25 wide1375](../notebooks/training_history/FSO_SC_stage2_error_range_to1_v25.ipynb): 같은 정책으로 351회 추가. Stage 2 누적 1,375회로, 이미지 투입량 기준 전체 데이터 1회분입니다.

이 오류 주입 정책은 독립 구현입니다. 저자의 학습 설정이나 실제 물리 채널의 오류 분포를 그대로 복원한 것은 아닙니다. 보관된 노트북에는 코드와 학습 이력이 남아 있지만, 원본 데이터·체크포인트·이전 진단 요약(예: v24에 필요한 v23 오류 범위 보고서)을 참조합니다. 따라서 새 데이터용 01–03 절차를 대신할 수 없습니다. v12 노트북에 포함된 별도의 Stage 2 실험은 v25의 학습 경로에 포함되지 않습니다.

## 공개 전 확인

- 당시 실험을 다시 실행하려면 원본 입력과 실행 폴더 전체를 Drive 또는 NAS에 비공개로 보관하세요.
- GitHub에는 이 공개용 소스 폴더만 올리세요. 모델 파일, 원본 데이터, 이미지별 결과 압축 파일, 인증 정보, 비공개 다운로드 링크를 커밋하지 마세요.
- 추후 학습 체크포인트나 데이터를 공개하려면 출처와 배포 권한을 별도로 확인하세요. 현재 공개본에는 포함되지 않습니다.
