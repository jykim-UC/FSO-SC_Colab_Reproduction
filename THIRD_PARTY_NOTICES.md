# Third-party components and attribution

This is an independent reproduction, not an official release from the FSO-SC paper's authors. This file records upstream sources; it does not assign a license to this repository or replace upstream license texts.

## Research and implementation references

- FSO-SC paper: Chen et al., *IEEE Transactions on Communications* 73(10), 9168–9183, 2025, [DOI 10.1109/TCOMM.2025.3562356](https://doi.org/10.1109/TCOMM.2025.3562356). Paper/code/weights are not bundled as the authors' release.
- [MoVQ](https://arxiv.org/abs/2209.09002), [MoVQGAN implementation/configuration](https://github.com/ai-forever/MoVQGAN), [Kandinsky MoVQ modules](https://github.com/ai-forever/Kandinsky-2/blob/main/kandinsky2/vqgan/movq_modules.py), and [VQGAN/taming-transformers](https://github.com/CompVis/taming-transformers) inform architecture and operational assumptions. Consult each upstream repository for its license and any attribution required for reused material.
- [PerceptualSimilarity / LPIPS](https://github.com/richzhang/PerceptualSimilarity) supplies perceptual evaluation/loss components. Package code, perceptual-network weights and backbone weights may have distinct upstream notices.
- [H. Jin, Caltech thesis, Chapter 7](https://thesis.caltech.edu/3191/1/HJinthesis.pdf) is a construction reference for RA codes. The seeded matrix here is not the FSO-SC paper's undisclosed matrix.
- [Pfau et al., blind phase search](https://doi.org/10.1109/JLT.2008.2010511) and [Meiyappan et al., coherent carrier estimation](https://doi.org/10.1109/JLT.2013.2260723) inform the receiver comparisons. The v34 block-affine 16QAM implementation is an independent engineering variant, not upstream code or the authors' receiver.

## BPG 0.9.8 — downloaded separately

The installer retrieves the [official BPG distribution](https://bellard.org/bpg/). No BPG executable, DLL, codec source archive, or compiled library is included in this repository.

The official BPG page states:

- The encoder as a whole, including x265, is GPL version 2; encoder sources excluding x265 are BSD-licensed.
- The decoder's modified FFmpeg code is LGPL version 2.1; the BPG decoding library excluding FFmpeg code is BSD-licensed.

Keep the license files from the exact downloaded archive with your local installation. Do not infer that a license selected later for this repository also covers BPG or its dependencies. The installer records archive/binary hashes but does not claim publisher-signature verification.

## Data and external model files

No ImageNet images, NAIP image collection, or trained semantic checkpoint is bundled. ImageNet access follows its [official distribution terms](https://www.image-net.org/download.php). NAIP provenance/attribution follows the [Earth Engine catalog](https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ), which requests acknowledgement of USDA Farm Production and Conservation - Business Center, Geospatial Enterprise Operations.

Dependency downloads, pretrained VGG/AlexNet backbones, and future checkpoint releases require their own provenance and license review. This notice does not grant redistribution rights to files absent from this release.

## Repository license status

No `LICENSE` is selected in this draft. Before a public release, the maintainer must confirm the repository's own authorship/license and retain any upstream notices applicable to actual reused code. Do not label the entire dependency chain MIT/BSD merely because the independent wrapper code receives such a license later.
