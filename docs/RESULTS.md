# 개발 및 최종 holdout 결과

이 문서는 공개 자체 데이터 워크플로(01–03)와 대응하는 **이전 baseline**을 보존합니다. 다른 가중치(v25 wide1375)로 수행한 [후속 v27·v31–v34 결과](LATEST_RESULTS.md)와 합산하거나 동일한 holdout으로 해석하지 마세요.

상태: **conditional/partial reproduction**. 기존 개발 200장과 별도 frozen holdout 200장을 구분해 보고합니다. 두 run 모두 clean 2개 + 2프로필×14조건×3방법의 86개 집계 행이며, 공개 JSON에는 원본 좌표 파일명·계정 경로를 포함하지 않습니다.

## 개발 결과: 기존 200장

개발 run `e23292642de7f418`의 `bpg_repaired_summary.json` 집계입니다. 이미 사용한 200장으로 최종 holdout 성능이 아니며, 기존 값은 `results/dev_summary.json`에 보존했습니다.

### 개발 프로토콜

- Source rate: FSO-SC와 BPG 평균 0.21875 bpp. BPG 총 358,400 bytes, padding 없이 평균 1,792 bytes/image.
- BPG 이미지별 rate: 0.156982–0.277344 bpp. 평균만 맞췄으며 per-image fixed rate 또는 동일 channel-use 비교가 아닙니다.
- Native BPG QP는 byte 수만으로 고정합니다. 두 BPG 분기와 모든 채널 조건에서 같은 매핑을 재사용하며 품질 metric으로 선택하지 않습니다.
- 14개 개발 조건: 30/33/36 dBm×QPSK/8QAM/16QAM(zenith 30°), 36 dBm×세 변조(60°), 36 dBm QPSK snow/fog(0°). 전체 논문 sweep이 아닙니다.
- 두 channel profile과 FSO-SC/BPG/BPG+LDPC를 보고합니다. seed 20261028, LPIPS AlexNet v0.1, 이미지 단위 bootstrap 2,000회입니다.
- 동일 seed/fading draw를 쓰지만 payload 길이가 달라 waveform·noise realization 전체가 완전히 동일한 것은 아닙니다.

### 개발 핵심 수치

값은 평균 LPIPS이며 낮을수록 좋습니다. BPG decode 실패에는 아래에 설명한 marker가 포함됩니다.

| 조건 | FSO-SC | BPG | BPG+LDPC |
|---|---:|---:|---:|
| Clean, no channel | 0.197261 | 0.391852 | — |
| Background-only, clear 30 dBm/16QAM/30° | 0.197261 | 0.391852 | 0.391852 |
| Coherent extension, clear 30 dBm/16QAM/30° | 0.224354 | 0.930816 | 0.397997 |
| Background-only, snow 36 dBm/QPSK/0° | 0.197261 | 0.391852 | 0.391852 |
| Coherent extension, snow 36 dBm/QPSK/0° | 0.895687 | 1.000000 | 1.000000 |

Coherent extension의 clear 30 dBm/16QAM 조건에서 BPG decode failure는 66%, BPG+LDPC는 0.5%였습니다. snow에서는 두 BPG 분기가 100% decode failure였습니다. 반면 background-only snow에서는 오류 없는 clean 수준입니다. 이는 **미공개 수신기 잡음 가정이 결론에 크게 영향을 준다**는 증거이지, 한 프로필이 실제 실험과 맞는다는 검증은 아닙니다.

BPG 실패 시 `lpips_mean`에는 paper-style marker 1을 넣습니다. gray replacement를 실제로 렌더링한 LPIPS는 별도 필드입니다. 표의 1.0을 실제 영상 LPIPS 또는 metric의 상한이라고 해석하면 안 됩니다. FSO-SC의 유한한 실패 복구 영상도 통계에서 제외하지 않습니다.

이 개발 run은 검증된 clean/non-8QAM FSO-SC 결과 일부를 이전 run에서 hash 확인 후 재사용했습니다. BPG와 8QAM 결과는 수리된 경로로 다시 계산했습니다. 요약 전체가 한 번의 새 평가로 생성된 것은 아닙니다.

## 최종 holdout: 별도 200장

Run `843f68538c469cd5`, protocol `fso_sc_final_holdout_v1`의 평가를 완료했습니다. 모델·선택 영상·코드·채널 설정을 고정하고 두 프로필과 모든 14조건을 평가했으며, 개발 점수는 재사용하지 않았습니다. 86행 모두 200장으로, 총 17,200개의 이미지×평가 행입니다. 서로 다른 이미지가 17,200장이라는 뜻은 아닙니다.

- 개발 200장과 파일명·파일 SHA-256 중복은 0입니다. 선택 seed 20261029, 채널 seed 20261028, LPIPS AlexNet v0.1과 이미지 단위 bootstrap 2,000회를 사용했습니다. 파일 중복 없음은 지리적·근접 영상 독립성 또는 checkpoint의 학습 이력 검증을 뜻하지 않습니다.
- 평가 코드 SHA-256: `2dd25e3b9686592500871ccc478c100286c9317b950f3156ffd5b8f357b83f08`. 모델·데이터·설정 hash와 전체 조건별 통계·95% CI는 `results/final_holdout_summary.json`에 기록했습니다.
- Source rate 평균은 FSO-SC/BPG 모두 0.21875 bpp입니다. Native BPG 총 358,400 bytes, 평균 1,792 bytes/image, 범위 1,272–2,231 bytes(0.155273–0.272339 bpp), source padding 0입니다. byte 수만으로 고정한 QP 매핑을 두 BPG 분기·모든 조건에서 재사용했습니다.
- QPSK/16QAM의 transmitted rate 평균은 FSO-SC/BPG 0.21875 bpp, BPG+LDPC 0.3078125 bpp입니다. 8QAM은 symbol padding으로 소폭 증가합니다. 따라서 동일 source rate 비교이며 **동일 channel-use budget 비교가 아닙니다.**
- `elapsed_this_cell_seconds`는 428.573초입니다. 최종 평가 셀의 시간이며 전체 학습·빌드·파일 복사 시간은 아닙니다.

평균 LPIPS(낮을수록 좋음)의 대표 행은 아래와 같습니다. BPG 실패 marker 정책은 개발 평가와 같습니다. 두 프로필의 전체 조건은 [최종 집계](../results/final_holdout_summary.json)와 [전체 곡선](../results/final_holdout_curves.png)에 함께 공개했습니다.

| 조건 | FSO-SC | BPG | BPG+LDPC |
|---|---:|---:|---:|
| Clean, no channel | 0.200522953 | 0.401726474 | — |
| Background-only, clear 30 dBm/16QAM/30° | 0.200522953 | 0.401726474 | 0.401726474 |
| Coherent extension, clear 30 dBm/16QAM/30° | 0.229094586 | 0.930625266 | 0.403332292 |
| Coherent extension, clear 36 dBm/16QAM/60° | 0.212309160 | 0.813735073 | 0.401726474 |
| Background-only, snow 36 dBm/QPSK/0° | 0.200522953 | 0.401726474 | 0.401726474 |
| Coherent extension, snow 36 dBm/QPSK/0° | 0.895135193 | 1.000000000 | 1.000000000 |
| Background-only, fog 36 dBm/QPSK/0° | 0.908061915 | 1.000000000 | 1.000000000 |
| Coherent extension, fog 36 dBm/QPSK/0° | 0.911324317 | 1.000000000 | 1.000000000 |

Coherent clear 30 dBm/16QAM/30°에서 BPG decode failure는 63.5%, BPG+LDPC는 0%였습니다. Coherent snow에서는 두 BPG 분기가 100% 실패했고, 수신 payload BER(LDPC는 post-FEC)은 FSO-SC/BPG/BPG+LDPC 각각 약 0.4976/0.4926/0.4940입니다. FSO-SC의 LPIPS가 marker 1보다 낮다는 사실만으로 유효한 영상 복구를 주장할 수 없습니다. 두 fog 프로필에서도 BER은 약 0.5로 붕괴합니다.

## 결과의 의미와 한계

Clean에서도 두 codec의 reconstruction 성능이 다릅니다. 최종 holdout의 clean MSE는 FSO-SC 0.010326352, BPG 0.006858331로 BPG가 낮습니다. LPIPS 우위를 모든 화질 metric의 우위로 일반화하거나, 채널 후 LPIPS 차이 전체를 통신상의 robustness 이득으로 돌릴 수 없습니다. source compression, 채널 오류, DSP 복구와 decode failure를 분리해서 읽어야 합니다.

원문과 실제로 다른 핵심 조건도 있습니다. [Chen et al.](https://doi.org/10.1109/TCOMM.2025.3562356)의 Sec. IV-D/Fig. 7은 snow 36 dBm/QPSK/0°에서 FSO-SC LPIPS < 0.4, BPG+LDPC는 1(복구 실패)을 보고합니다. 이 구현은 coherent 프로필에서 FSO-SC 복구를 재현하지 못하고, background-only에서는 BPG+LDPC도 복구되어 **어느 프로필도 이 공동 결과를 동시에 재현하지 못했습니다.** Fig. 4(a)의 clear 30 dBm/16QAM/30°에서도 논문의 BPG+LDPC는 1인 반면 이 구현은 0.403332292입니다. 일부 우열 경향만 부합하며, 논문 수치가 대략 일치한다고 볼 근거는 없습니다.

오류 없는 background-only clear 조건에서는 전력에 따른 곡선 변화가 거의 없습니다. 이 구간으로 3 dB gain을 측정할 수 없습니다. 현재 sparse power sweep, 대체 데이터, 미공개 원본 학습 설정과 수신기 가정으로는 [원 논문](https://doi.org/10.1109/TCOMM.2025.3562356)의 약 3 dB 결과를 독립적으로 확인했다고 말할 수 없습니다.

## 아직 하지 않은 것

- 저자 원본 이미지·가중치·LDPC parity-check matrix·VPI 설정과의 일치 검증.
- 전체 논문 figure/sweep 및 실제 광학·FPGA/10 Gbps 실험 재현.
- 동일 channel-use budget의 추가 비교, 채널 파라미터 불확실성에 대한 충분한 sensitivity sweep.

최종 holdout을 개발 표와 분리해 보존했습니다. 결론에 맞춰 프로필이나 실패 정책을 바꾸지 마세요.
