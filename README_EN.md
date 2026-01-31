
<div align="right">

[中文](README.md) ｜ **English**

</div>

![](assets/qianyan.png)
![](assets/lolv1.gif)
![](assets/lsrw.gif)
![](assets/DarkFace.gif)
![](assets/NonPair.gif)

# Consumer-Grade Embodied Robots Sensing and Computing Integrated Low-Light Image Enhancement for Perception Robustness

> Consumer-grade embodied robots (e.g., smart-home and service robots) require robust onboard perception under adverse illumination, while operating with constrained compute, memory, and power budgets. In low-light scenes, sensor noise, color shifts, and detail loss are easily exacerbated by RGB-space enhancement where luminance, contrast, and chroma are implicitly coupled, causing over-exposure, color bias, and reduced perception robustness for downstream tasks. This paper presents BCC-LabNet, a deployment-oriented low-light enhancement method in the CIE-Lab space that explicitly decouples brightness, contrast, and chroma via three lightweight branches, and introduces an FWBlock for targeted noise suppression in challenging scenes. The factorized design provides interpretable, component-wise corrections and improves cross-camera and cross-environment generalization without per-domain retraining. We evaluate image quality using PSNR/SSIM/LPIPS on paired benchmarks and NIQE/BRISQUE/UICM on unpaired real-world images, and quantify perception robustness using low-light object detection metrics (e.g., mAP) under fixed detector settings. Experiments on paired, unpaired, and cross-domain benchmarks show competitive restoration quality and consistent gains in downstream detection, indicating more stable task-relevant structures and colors. These results suggest that explicit Lab-space decoupling is a practical and reliable enhancement front end for perception robustness in consumer-grade embodied robots.。

![](overview.png)

---

## Contents
- [Environment](#environment)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Validation & Visualization](#validation--visualization)
- [Testing / Metrics](#testing--metrics)
- [Ablation Studies](#ablation-studies)
- [Efficiency Comparison](#efficiency-comparison)
- [Comparative Results](#comparative-results)
- [Generalization Experiments](#generalization-experiments)
- [Synthetic Low‑Light Data (Optional)](#synthetic-low-light-data-optional)
- [Loss Design](#loss-design)
- [Dataset Links](#dataset-links)
- [License](#license)
- [Project Tree](#project-tree)
- [References & Acknowledgements](#references--acknowledgements)

---

## Environment

```bash
conda env create -f BCC.yml
conda activate bcc
```

> Key deps: PyTorch, `pytorch-msssim`, `opencv-python`, `pyiqa` (NIQE), `torchvision`.

---

## Data Preparation

Paired data (low/normal). Example:

```
data/
 └─ LOLv1/
    ├─ Train/
    │  ├─ input/
    │  └─ target/
    └─ Test/
       ├─ input/
       └─ target/
```

Other sets (LoLI‑Street, LOLv2) are supported—configure paths in `train.py` / `test.py`.

---

## Training

1) Set `train_low`, `train_high`, `test_low`, `test_high` in `train.py`.  
2) Run:

```bash
python train.py
```

- Default **256×256 crops**, `batch_size=1`; **AdamW** (lr `2e-4`) + **OneCycleLR**;  
- PSNR/SSIM on val during training; supports loading checkpoints (`strict=False`).

---

## Validation & Visualization

`val.py` shows enhanced RGB, `ΔL`, attention, illumination/reflection; histogram/gamma views; one‑click save. Use `get_attention_with_hook` to inspect intermediate tensors.

---

## Testing / Metrics

```bash
python test.py
```

Saves outputs and **PSNR/SSIM**; includes **NIQE**; for low‑only use `LowOnlyDataset` + NIQE.

---

## Ablation Studies

Ablations on **LOLv2‑Real**.

![Ablation](assets/fig_7.png)

> BCC is channel‑uncompressed; baseline is three‑branch only; **no GT‑Mean**.

---

## Efficiency Comparison

**3.17M params / 3.76G FLOPs**, competitive quality with lower compute than many peers.

![Complexity](assets/fig_9.png)

> **No GT‑Mean**.

---

## Comparative Results

LOL‑v1 / LOL‑v2‑Real / VisDrone / LSRW‑Huawei quantitative comparison.

![Comparison](assets/fig_8.png)

> **No GT‑Mean**.

---

## Generalization Experiments

Five unpaired no‑ref datasets (DICM, LIME, MEF, NPE, VV); under the same settings **NIQE = 4.1215**, **UICM = 0.9776**.

![](assets/fig_10.png)

> **No GT‑Mean**.

Cross‑dataset: train on LOL‑v1, test on 200 LoLI‑Street images without fine‑tuning.

![](assets/fig_11.png)

> **No GT‑Mean**.

---

## Synthetic Low‑Light Data (Optional)

`synthesize_low_light.py`: simple darkening + noise, or realistic non‑uniform darkening with factor maps saved.

---

## Loss Design

`CombinedLoss`: Smooth‑L1, VGG‑19 perceptual, histogram, PSNR penalty, Color/Mean constraints, optional MS‑SSIM.

```bash
Baidu: https://pan.baidu.com/s/1gTVcG7pbiOWWtay-7fkzZw?pwd=79sa  Code: 79sa
Google Drive: https://drive.google.com/drive/folders/1JlGYF8-zxhlJAb6Sp0yH3B5qlQvrd5EJ?usp=sharing
```

---

## Dataset Links

```bash
https://pan.baidu.com/s/1gikbndlP69_j0hXJ4MmUbw?pwd=ntmx  Code: ntmx
```

---

## License

Academic use by default; contact authors for commercial use/redistribution.

---

## Project Tree

```
.
├── BCC.yml
├── train.py / val.py / test.py
├── dataloader.py
├── losses.py
├── synthesize_low_light.py
└── overview.png
```

---

## References & Acknowledgements

We acknowledge prior low‑light/Retinex work: LIME, NPE, Retinex series, EnlightenGAN, Kindling the Darkness, Deep Retinex Decomposition, etc.
