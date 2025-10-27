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
