# 데이터와 외부 가중치

이 문서는 자체 데이터로 실행하는 01–03번 노트북 기준입니다. 후속 고정 실험에 필요한 원본 입력은 [별도 재실행 안내](REPLAY.md)에 있습니다.

데이터·모델 checkpoint·BPG 실행파일은 공개 번들에 포함하지 않습니다. 계정 인증정보, Drive file ID, 원본 위경도 파일명 목록도 포함하지 않습니다.

## 학습 입력

Colab의 기본 입력 위치는 자신의 `/content/drive/MyDrive/FSO_data`입니다. notebook 설정에서 바꿀 수 있습니다.

| 파일 | 예상 이미지 수 | 역할 |
|---|---:|---|
| `imagenet_train_10000.zip` | 10,000 | ImageNet subset 학습 |
| `GEE_train_fixed.zip` | 1,000 | NAIP proxy 학습 |
| `GEE_test.zip` | 1,000 | 학습과 분리된 평가 후보 pool |

ZIP에는 PNG/JPG/JPEG 이미지가 있어야 합니다. 학습은 RGB 변환·최대 중앙 정사각 crop·bicubic 256×256·[-1,1] 정규화를 적용합니다. test 데이터를 학습 또는 checkpoint 선택에 넣지 마세요.

원 논문은 ImageNet 10,000장과 Google Earth 원격탐사 영상 1,000장을 사용하지만, 저자 원본 이미지 목록은 이 구현에 제공되지 않았습니다. 이 저장소의 NAIP 영상은 **동일 데이터가 아닌 대체 데이터**입니다. 데이터 장수 일치가 분포 또는 split 일치를 뜻하지 않습니다.

`notebooks/01_prepare_GEE.ipynb`는 위 세 ZIP의 이미지 수, SHA-256과 데이터 간 중복을 검사하는 **audit 전용** notebook입니다. 영상을 다운로드하거나 수리하지 않으며 Earth Engine API도 호출하지 않습니다. 사용할 권한이 있는 ZIP은 사용자가 별도로 준비해야 합니다. 저자 원본 원격탐사 데이터의 전체 수집 절차·manifest가 제공되지 않았으므로 정확히 같은 데이터의 자동 재구성을 보장하지 않습니다.

## 직접 준비할 때 기록할 항목

ImageNet은 [공식 배포·접근 안내](https://www.image-net.org/download.php)를 따라 준비합니다. NAIP는 [USDA/NAIP/DOQQ Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ)를 참고하세요. catalog는 USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations에 대한 attribution을 요청합니다.

`tools/prepare_imagenet_subset.py`는 별도로 확보한 원본 ImageNet archive에서 subset을 만드는 보조 도구입니다. source/output 경로를 CLI 인자로 전달하며 원본 이미지를 배포하거나 계정 접근을 대신하지 않습니다. `python tools/prepare_imagenet_subset.py --help`로 입력 형식을 확인하세요.

재수집 시 collection/version, 날짜 filter, 영역·샘플링 seed, RGB bands, 공간해상도, crop/resize, 결측 제거, 중복 제거 방법과 ZIP SHA-256을 별도 manifest로 저장하세요. 계정·quota·영상 갱신에 따라 재수집 결과가 달라질 수 있습니다. 원본 manifest가 공개되지 않은 상태에서 동일 픽셀 재현을 보장하지 않습니다.

학습·개발·holdout 사이에 최소한 파일 SHA-256 중복 검사를 하세요. 파일 중복 검사는 공간 중첩이나 인접 지역의 상관까지 배제하지 않습니다. 그런 확인 없이 geographic independence를 주장하지 마세요.

## 개발과 holdout

기존 개발 200장과 별도 frozen holdout 200장의 최종 평가를 완료했으며, 각각 `results/dev_summary.json`과 `results/final_holdout_summary.json`으로 분리했습니다. 최종 run `843f68538c469cd5`는 개발 영상과 파일명·SHA-256 중복이 0이고, 개발 점수를 재사용하지 않고 86개 집계 행 모두 200장씩 평가했습니다. 선택은 결과를 보기 전 한 번 고정했습니다.

최종 holdout은 평가 pool 1,000장에서 개발 200장을 제외한 800장 중 파일 hash 순위와 seed 20261029로 선택했습니다. 제공된 NAIP 학습 ZIP 1,000장과도 파일 SHA-256 중복이 0임을 확인했지만, 이는 geographic/near-image independence나 checkpoint의 전체 학습 이력을 입증하지 않습니다. `notebooks/03_final_holdout.ipynb`에서도 평가 전에 selection과 모델·코드 hash를 고정하고, 결과를 본 후 채널 프로필·LDPC matrix·QP 정책·checkpoint를 고르지 마세요.

공개 기본값은 `DATA_MODE='own_data'`, `ENFORCE_REFERENCE_MODEL=False`, `RUN_NAME='paper_faithful_public_run1'`입니다. `02_train.ipynb`에서도 같은 실행명을 사용하세요. 자신의 새 평가 corpus에서 metric 계산 전에 hash 순위로 개발용 200장을 예약하고, 나머지에서 holdout 200장을 선택합니다. 두 채널 프로필·14개 조건은 reference 절차와 같지만 데이터·모델이 다르므로 논문 또는 보고된 개발·최종 결과와 같은 숫자를 주장할 수 없습니다.

`reference_locked` 모드는 기존 비공개 reference 데이터·모델을 보유한 경우를 위한 엄격한 검증 경로입니다. `config/holdout_manifest.json`에는 원본 파일명 대신 hash만 기록합니다. 그 manifest가 이미지·checkpoint를 제공하는 것은 아니며, 공개 기본 설정은 reference 모델 hash 일치를 강제하지 않습니다.

## checkpoint와 보안

보고된 학습 checkpoint는 현재 공개 배포하지 않습니다. 자기 실행에서 학습하거나, 향후 출처와 SHA-256이 함께 공개된 checkpoint를 사용해야 합니다. checkpoint가 없으면 저장소의 평가 코드를 실행할 수 있어도 기존 숫자를 그대로 복원할 수 없습니다.

PyTorch checkpoint는 pickle을 포함할 수 있으므로 신뢰하지 않는 파일을 로드하지 마세요. 학습 재개는 모델뿐 아니라 optimizer/RNG/진행 위치가 같은 실행의 checkpoint를 사용합니다. final weights만으로는 정확한 중단 지점 재개가 되지 않습니다.

VGG/AlexNet 계열의 perceptual model weights 등은 실행 중 dependency가 별도로 다운로드할 수 있습니다. 그 파일의 배포 조건과 출처는 각 upstream을 따릅니다.
