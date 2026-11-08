# 외부 구성요소와 출처 안내

[English](THIRD_PARTY_NOTICES.md) · [한국어 README](README.ko.md)

이 저장소는 FSO-SC 논문의 독립 재현이며, 논문 저자의 공식 배포본이 아닙니다. 이 문서는 외부 구성요소의 출처를 정리합니다. 저장소 자체에 라이선스를 부여하거나 원본 라이선스 문서를 대신하지 않습니다.

## 참고 연구와 구현

- FSO-SC 논문: Chen 외, *IEEE Transactions on Communications* 73(10), 9168–9183, 2025, [DOI 10.1109/TCOMM.2025.3562356](https://doi.org/10.1109/TCOMM.2025.3562356). 저자의 논문 PDF·공식 코드·가중치를 함께 배포하지 않습니다.
- [MoVQ](https://arxiv.org/abs/2209.09002), [MoVQGAN 구현과 설정](https://github.com/ai-forever/MoVQGAN), [Kandinsky의 MoVQ 모듈](https://github.com/ai-forever/Kandinsky-2/blob/main/kandinsky2/vqgan/movq_modules.py), [VQGAN/taming-transformers](https://github.com/CompVis/taming-transformers)를 모델 구조와 구현 가정의 참고 자료로 사용했습니다. 원본 코드의 라이선스와 재사용 시 필요한 출처 표기는 각 저장소를 확인하세요.
- [PerceptualSimilarity / LPIPS](https://github.com/richzhang/PerceptualSimilarity)는 지각 품질 평가와 학습 손실에 쓰입니다. 패키지 코드, 지각 평가용 네트워크 가중치, 백본 가중치에는 서로 다른 이용 조건이 적용될 수 있습니다.
- [H. Jin의 Caltech 학위논문 7장](https://thesis.caltech.edu/3191/1/HJinthesis.pdf)은 RA 코드 구성의 참고 자료입니다. 이 저장소에서 고정 시드로 만든 행렬은 FSO-SC 논문의 미공개 행렬이 아닙니다.
- [Pfau 외의 블라인드 위상 탐색](https://doi.org/10.1109/JLT.2008.2010511)과 [Meiyappan 외의 코히런트 반송파 추정](https://doi.org/10.1109/JLT.2013.2260723)을 수신기 비교에 참고했습니다. v34의 16QAM 블록 단위 위상 추적은 독립 구현이며, 참고 자료의 코드나 FSO-SC 논문의 수신기를 그대로 가져온 것이 아닙니다.

## BPG 0.9.8 — 별도 다운로드

설치 스크립트는 [공식 BPG 배포처](https://bellard.org/bpg/)에서 파일을 가져옵니다. 이 저장소에는 BPG 실행 파일, DLL, 코덱 소스 압축 파일, 컴파일된 라이브러리가 포함되지 않습니다.

공식 BPG 페이지의 안내는 다음과 같습니다.

- x265를 포함한 인코더 전체에는 GPL 2가 적용됩니다. x265를 제외한 인코더 소스는 BSD 라이선스입니다.
- 디코더의 수정된 FFmpeg 코드에는 LGPL 2.1이 적용됩니다. FFmpeg를 제외한 BPG 디코딩 라이브러리는 BSD 라이선스입니다.

설치할 때 내려받은 배포본의 라이선스 파일도 보관하세요. 이 저장소에 나중에 부여하는 라이선스가 BPG나 그 의존성에도 적용된다고 해석하면 안 됩니다. 설치 스크립트는 압축 파일과 바이너리 해시를 기록하지만, 배포자의 서명을 검증하지는 않습니다.

## 데이터와 외부 모델 파일

ImageNet 이미지, NAIP 이미지 모음, 학습된 시맨틱 모델 체크포인트는 포함하지 않습니다. ImageNet은 [공식 배포 조건](https://www.image-net.org/download.php)을 따릅니다. NAIP의 출처 및 표기 방법은 [Earth Engine 카탈로그](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ)를 참고하세요. 카탈로그는 USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations를 출처로 표시하도록 안내합니다.

의존성 다운로드, 사전학습된 VGG/AlexNet 백본, 향후 체크포인트 배포는 각각 출처와 이용 조건을 별도로 확인해야 합니다. 이 문서는 공개본에 없는 파일의 재배포 권한을 부여하지 않습니다.

## 저장소 자체 라이선스

이 저장소에는 아직 `LICENSE`를 지정하지 않았습니다. 공개 배포 전에는 직접 작성한 부분의 저작권과 라이선스를 확인하고, 실제로 재사용한 외부 코드에 적용되는 고지를 유지하세요. 자체 작성 코드에 MIT나 BSD를 선택하더라도 모든 의존성이 같은 라이선스가 되는 것은 아닙니다.
