![](assets/lolv1.gif)
![](assets/lsrw.gif)
![](assets/DarkFace.gif)
![](assets/NonPaired.gif)
# BCC‑LabNet：低光照图像增强（Lab & Retinex）

> 本仓库包含 **BCC‑LabNet** 的训练、验证、测试与数据合成脚本，用于在低光照场景下对图像进行增强与细节恢复。模型在 Lab 颜色空间进行建模，并结合 Retinex 思想与注意力模块以提升亮度与色度一致性。

![](overview.png)

---

## 目录
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练 Train](#训练-train)
- [验证与可视化 Val](#验证与可视化-val)
- [测试 Test / 指标评估](#测试-test--指标评估)
- [合成低光数据（可选）](#合成低光数据可选)
- [损失函数设计](#损失函数设计)
- [参考与致谢](#参考与致谢)

---

## 环境准备

推荐使用 Conda 根据 `BCC.yml` 一键创建环境：

```bash
conda env create -f BCC.yml
conda activate base
```

> 关键依赖：PyTorch、`pytorch-msssim`、`opencv-python`、`pyiqa`（用于 NIQE 指标）、`torchvision` 等。

---

## 数据准备

脚本默认采用 **成对数据**（低光/正常曝光），典型目录结构示例：

```
data/
 └─ LOLv1/
    ├─ Train/
    │  ├─ input/   # 低光训练图
    │  └─ target/  # 正常曝光训练图
    └─ Test/
       ├─ input/   # 低光测试图
       └─ target/  # 正常曝光测试图
```

也支持如 **LoLI‑Street**、**LOLv2** 等数据集，只需将 `*_low` 与 `*_high/target/Normal` 路径按需填写（参见 `train.py` / `test.py` 中示例注释）。

---

## 训练 Train

1. 在 `train.py` 顶部按需填写/修改数据路径：

   - `train_low`, `train_high`：训练集低光/正常曝光目录  
   - `test_low`, `test_high`：验证集低光/正常曝光目录

2. 可直接运行：

```bash
python train.py
```

- DataLoader 默认 **随机裁剪 256×256**、`batch_size=1`；优化器使用 **AdamW**，学习率 `2e-4`，并配合 **OneCycleLR** warmup 调度。  
- 训练过程中会在验证集上计算 PSNR/SSIM 并打印。  
- 代码支持加载一个预训练权重作为初始化（`strict=False`），可按需替换路径。

## 验证与可视化 Val

`val.py` 包含可视化与诊断函数，帮助分析：
- 增强后的 RGB、`ΔL`（亮度增量）、注意力图、以及分解的照明/反射量等；
- 多种 `ΔL` 归一化与直方图对比、伽马校正可视化；
- 一键保存多路可视化到本地。

> 你可以在推理后调用这些函数，或参考 `get_attention_with_hook` 的用法把模型中间量“钩”出来进行可视化。


## 测试 Test / 指标评估

将测试集路径与权重文件在 `test.py` 中按需填写，然后：

```bash
python test.py
```

- 程序会保存增强结果图，并计算 **PSNR / SSIM**；  
- 也提供 **NIQE** 无参考指标的计算函数（基于 `pyiqa`），可在无 GT 的场景使用；  
- 仅低光输入（无 GT）时，使用 `LowOnlyDataset` 与 NIQE 评估流程。
## 消融实验

为验证各核心模块的有效性，我们在LOLv2-Real数据集上进行了系统的消融实验。

![表1. 在LOL-v2-Real数据集上的消融实验](assets/fig_7.png)

请注意，BCC模型为通道未压缩模型，基线模型（第一行）为仅包含三子网解耦的结构。我们没有使用GT-Mean策略。
## 运行效率对比

当前低光照增强领域呈现出两种主要趋势：其一是以LYT-Net为代表的极致轻量化设计，它们以极低的参数量和计算量，但是性能较差；其二是如RetinexMamba等模型，为追求性能上限而采用了较高的计算复杂度。在此背景下，BCC-LabNet采取了一条平衡的折中路线：其参数量(3.17M)虽高于部分轻量级模型，但其计算量(3.76G Flops)显著低于多数同等性能的主流模型(如Wave-Mamba、CWNet等），甚至低于参数量更少的UHDFour。这一结果表明，本模型通过其Lab空间解耦与Retinex协同的架构设计，实现了计算资源的有效利用，在以适中参数规模获得竞争力的增强效果的同时，具备了更高的性能、更优的部署潜力与更低的计算量。
![表3. 模型复杂度对比](assets/fig_9.png)

请注意，我们没有使用GT-Mean策略。

## 对比实验

表2. 在LOL-v1、LOL-v2-Real、VisDrone和LSRW-Huawei数据集上的定量比较。最佳和次佳结果分别用粗体和下划线标出。请注意，我们没有使用GT-Mean策略。其中在LOL-v1、LOL-v2-Real、LSRW-Huawei数据集中BCC模型为通道未压缩模型，VisDrone为通道压缩模型。

![表2. 在LOL-v1、LOL-v2-Real、VisDrone和LSRW-Huawei数据集上的定量比较](assets/fig_8.png)

请注意，我们没有使用GT-Mean策略。

## 泛化性实验



本节评测采用五个公开、广泛使用的非配对/无参考低照度图像数据集：DICM[9]、LIME[10]、MEF[11]、NPE[13]、VV，覆盖室内/室外、街夜景、背光与高动态范围等真实暗光场景，具有明显的噪声、色偏与对比度不足等退化特征，分辨率与纵横比多样；其中，DICM包含多设备采集的室内/室外暗光照片，照度分布广、失真类型多样，常用于检验总体鲁棒性；LIME侧重典型低照度与非均匀照明场景，光照变化剧烈、颜色先验复杂，常作为增强算法基准；MEF聚焦多曝光/曝光变化条件，含强背光与高动态范围样例，适于检验细节恢复与局部对比；NPE 强调自然感知质量，包含明显色偏与噪声，用于评估自然度保持与色彩还原；VV覆盖极暗/低可见度与复杂光源条件（如霓虹、雨夜、雾天等），强调在恶劣条件下的稳定性。
在真实场景的非配对设置下，我们采用两项无参考指标评估增强质量：NIQE（越低越好）与 UICM（越高越好，衡量色彩丰富度与自然度）。在相同推理设置下，BCC-LabNet取得 NIQE = 4.1215、UICM = 0.9776，表明模型在感知质量与色彩自然度上均表现优异。NIQE 的降低意味着纹理与结构更加自然、伪影更少；UICM 的提高则表明在提升亮度的同时避免了常见的色偏与过饱和。

![](assets/fig_10.png)

请注意，我们没有使用GT-Mean策略。



本实验评估各方法在跨数据集场景下的泛化能力。所有模型均在LOLv1数据集上训练，然后在LoLI-Street[23]数据集中随机选择的200张图像上进行测试，无需任何微调或额外训练。

![](assets/fig_11.png)

请注意，我们没有使用GT-Mean策略。

## 合成低光数据（可选）

`/synthesize_low_light.py` 提供两种批量生成方式：

- `batch_simulate_low_light(input_dir, output_dir, dark_factor_range=(0.1, 0.3))`：简单暗化 + 轻微噪声；
- `batch_realistic_simulation(input_dir, output_dir, dark_factor_range=(0.15, 0.4))`：更逼真的非均匀照明暗化（更暗），并保存暗化因子映射。

直接运行脚本后按提示选择模式即可。生成的图像与 `dark_factor_mapping.{json,txt}` 会保存在输出目录。是VisDrone数据集的合成代码


## 损失函数设计

训练采用 **加权多项损失**（`CombinedLoss`）：

- Smooth‑L1（像素一致性）  
- VGG‑19 感知损失（细节保持）  
- 直方图分布损失（整体亮度/对比度统计一致）  
- PSNR 相关惩罚（鼓励更高信噪比）  
- Color/Mean 约束（色彩/亮度均值一致性）  
- （可选）MS‑SSIM 结构一致性

默认权重以及相关输出已在下方中给出下载地址，实际训练可按数据与任务做微调。
```bash

百度云分享的文件：BCC-Files
链接: https://pan.baidu.com/s/1gTVcG7pbiOWWtay-7fkzZw?pwd=79sa 提取码: 79sa

---
google：https://drive.google.com/drive/folders/1JlGYF8-zxhlJAb6Sp0yH3B5qlQvrd5EJ?usp=sharing

```
## 参考与致谢

本项目实现参考了低光增强与 Retinex 相关工作，感谢以下研究：LIME、NPE、Retinex 系列、EnlightenGAN、Kindling the Darkness、Deep Retinex Decomposition等（详见论文汇总文档）。
## 数据集相关下载

```bash
链接: https://pan.baidu.com/s/1gikbndlP69_j0hXJ4MmUbw?pwd=ntmx 提取码: ntmx 

```


## 许可证

如果没有特殊说明，默认以学术研究目的使用。商用/再发布请先与作者沟通。


## 目录说明

```
.
├── BCC.yml                      # Conda 环境
├── train.py / val.py / test.py  # 训练 / 可视化 / 测试
├── dataloader.py                # 数据加载与裁剪
├── losses.py                    # 多项组合损失
├── synthesize_low_light.py      # 生成低光数据（可选）
└── overview.png                 # 模型/流程示意图
```


