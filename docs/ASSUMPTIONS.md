# 명시값, 독립 구현, 미결정 사항

아래 학습 예산·Stage 2 설정은 01–03번 **이전 baseline** 기준입니다. 후속 실험의 v25 wide1375는 [별도 학습 경로](REPLAY.ko.md#how-the-frozen-model-was-produced)를 사용합니다. v34는 고정 가중치·채널에서 16QAM 위상 추정만 교체한 선택적 비교입니다.

기준은 [Chen et al., TCOM 2025](https://doi.org/10.1109/TCOMM.2025.3562356)의 모델·손실·채널 설명입니다. 아래의 추정값은 저자 설정으로 확인된 값이 아닙니다.

## 모델과 학습

| 구분 | 구현과 근거 |
|---|---|
| 논문에서 보존한 항목 | RGB, Fig. 2/3의 3회 downsample/upsample과 4차원 양자화 특징, MSE + 0.1×VGG perceptual + VQ 항, Adam 1e-4/(0.5, 0.9), 학습 이미지 수 10,000+1,000 |
| 네트워크 해석 | Fig. 2/3을 바탕으로 독립 구현. decoder의 차원·순서가 불명확한 부분은 MoVQ/VQGAN식 일관된 배열로 해석; 동일 원본 구현을 보증하지 않음 |
| 입력·codebook 선택 | 256×256 입력, K=16,384와 14-bit index는 본 구현 선택. 3회 downsample일 때 14/64=0.21875 bpp로 논문의 약 0.22 bpp에 맞춘 것이며, 입력 크기·K가 논문에 명시되었다는 뜻이 아님 |
| 전처리 | 최대 중앙 정사각 crop → PIL bicubic 256, `reducing_gap=1` → RGB [-1,1]. augmentation 없음 |
| 수치 설정 | FP32, effective batch 8(현재 A100 기본 microbatch 4×2 accumulation), gradient value clipping 1. activation checkpointing은 기본 OFF. 메모리에 따라 microbatch는 조정 가능 |
| 학습 예산 | Stage 1/2 각각 초기 100 epochs. 논문이 명시한 epoch 또는 수렴 기준으로 주장하지 않음 |
| Stage 2 | 실제 FSO waveform 학습이 아닌 14-bit index-corruption surrogate. reference q∈{0,1e-4,1e-3,5e-3,1e-2}; q는 측정 채널 BER가 아님 |
| 추가하지 않은 기법 | GAN 학습, EMA, semantic 사전학습 가중치, codebook reset, 별도 LR schedule |
| checkpoint 선택 | `best`는 완료 epoch의 최소 **학습** LPIPS이며 validation best가 아님. 개발·최종 holdout 평가는 같은 Stage 2의 최종 100-epoch 모델을 사용 |
| 데이터 차이 | 저자 원본 원격탐사 이미지 대신 NAIP proxy. 원본 목록·원본 split과 동일하지 않음 |

전처리·FP32/batch 설정의 참고는 [공개 MoVQ 270M 설정](https://github.com/ai-forever/MoVQGAN/blob/main/configs/movqgan_270M.yaml)과 [data loader](https://github.com/ai-forever/MoVQGAN/blob/main/movqgan/data/dataset.py)입니다. [MoVQ 논문](https://arxiv.org/abs/2209.09002)은 spatially conditional normalization을, [VQGAN 논문](https://arxiv.org/abs/2012.09841)은 CNN 기반 VQ 표현을 설명합니다. 이 참고 구현들이 FSO-SC 저자의 원본 코드라는 뜻은 아닙니다.

## 광학 채널과 수신기

두 프로필을 함께 공개하며, 평가 전 선택을 기록합니다. 어느 쪽도 VPI 전체 재현 또는 물리적 우월성을 입증하지 않습니다.

| 프로필 | 포함하는 잡음 | 해석 |
|---|---|---|
| `paper_eq3_4_background_only` | background | 논문 Eq. (3)–(4)의 background-light 표기에 맞춘 식 수준의 이상화. shot/ASE가 없어 완전한 coherent receiver가 아님 |
| `coherent_receiver_extension_v6` | background + transmitter EDFA ASE + shot | coherent receiver 확장 해석. 논문에 공개되지 않은 수신기 설정까지 확인한 것은 아님 |

공통 미결정 사항은 optical background power→complex matched-filter variance 변환, FOV 단위, optical filter 폭, Málaga beta/phase, fading coherence, EDFA bandwidth/launch 기준과 shot-noise bandwidth/polarization입니다. Málaga의 생략 파라미터에는 구현 가정을 쓰고 moment residual을 기록합니다. 결과 곡선에 맞춘 parameter fitting은 하지 않습니다.

수신기는 symbol-rate matched-filter 근사입니다. 실제 phase를 넘기지 않고 주파수·위상 복구를 수행하지만, perfect frame boundaries/file length, block magnitude와 noise PSD를 아는 것으로 가정합니다. ADC·timing·IQ imbalance·실제 광학 하드웨어 전체는 구현하지 않습니다. rectangular 8QAM의 ML/BPS 처리 역시 독립 DSP이며 미공개 VPI 설정과 동일하다고 주장하지 않습니다.

## BPG, LDPC, 평가

- BPG 0.9.8의 실제 encoded bytes를 전송합니다. JPEG/가상 codec으로 대체하지 않습니다.
- source-rate matching은 native integer QP 후보의 encoded byte 수와 이미지 식별자만 사용합니다. LPIPS·SNR·채널 결과를 QP 선택에 쓰지 않습니다. dataset mean만 맞추며 per-image 고정 rate가 아닙니다.
- LDPC는 n=2,048/k=1,536(rate 3/4)의 seeded systematic RA construction입니다. 논문의 parity-check matrix가 아닙니다. normalized min-sum, 최대 50회, normalization 0.8을 사용합니다. RA 계열 근거는 [H. Jin의 학위논문 Chapter 7](https://thesis.caltech.edu/3191/1/HJinthesis.pdf)이며, 이 구체적인 matrix seed는 본 구현 선택입니다.
- 정확한 file length·framing을 가정합니다. CRC/ARQ가 없고 syndrome convergence를 CRC로 취급하지 않습니다. FEC block 및 QAM padding을 channel-use 계산에 포함합니다.
- 평가 LPIPS는 [공식 구현](https://github.com/richzhang/PerceptualSimilarity)의 AlexNet v0.1입니다. 학습 perceptual loss의 VGG와 다르며, 논문 평가 backbone과 동일하다고 단정하지 않습니다.
- bootstrap은 이미지 단위입니다. 여러 채널 실현이 있으면 이미지 내 평균 후 resampling합니다. 개발·최종 결과의 CI가 모델·데이터셋·채널 가정 불확실성까지 포함하지는 않습니다.
- BPG decode failure의 marker 1과 gray replacement의 실제 LPIPS를 별도 집계합니다. marker 1은 LPIPS의 수학적 상한이 아닙니다. 실패 프레임을 버려 평균을 개선하지 않습니다.

이 차이들 때문에 clean reconstruction 우위는 곧바로 전송전력 이득이 아니며, sparse sweep만으로 논문의 약 3 dB 주장을 확인할 수 없습니다.
