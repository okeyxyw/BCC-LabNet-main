# 使用钩子获取注意力图的替代方法
from PIL import Image

from base.BCCCOPYv2 import BCC_LabNet
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib

matplotlib.use('tkagg')
import torch.nn.functional as F
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_attention(attention_map,original_image=None ,L=None, save_path=None):
    """
    可视化注意力图

    Args:
        attention_map: 注意力图 [B, 1, H, W] 或 [B, H, W]
        original_image: 原始图像 [B, 3, H, W] (可选)
        save_path: 保存路径 (可选)
    """
    print(f"Input attention map shape: {attention_map.shape}")

    # 确保注意力图是4D的 [B, 1, H, W]
    if attention_map.dim() == 2:
        attention_map = attention_map.unsqueeze(0).unsqueeze(0)  # [H, W] -> [1, 1, H, W]
    elif attention_map.dim() == 3:
        attention_map = attention_map.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

    batch_size = attention_map.shape[0]

    if original_image is not None:
        print(f"Original image shape: {original_image.shape}")

    for i in range(batch_size):
        fig = plt.figure(figsize=(12, 5))

        # 获取当前样本的注意力图 [1, H, W]
        att = attention_map[i]  # [1, H, W]
        print(f"Single attention map shape: {att.shape}")

        # 调整注意力图大小到 [1, 1, 600, 400]
        # att_resized = F.interpolate(
        #     att.unsqueeze(0),  # [1, H, W] -> [1, 1, H, W]
        #     size=( 600,800,),
        #     mode='bilinear',
        #     align_corners=False
        # )
        att = att.squeeze(0).squeeze(0)  # [1, 1, 600, 400] -> [600, 400]
        print(f"Resized attention map shape: {att.shape}")

        if original_image is not None:
            # 原始图像处理
            img = original_image[i]  # [3, H, W]

            # 调整原始图像大小以匹配注意力图
            img_resized = F.interpolate(
                img.unsqueeze(0),  # [3, H, W] -> [1, 3, H, W]
                size=( 400,600,),
                mode='bilinear',
                align_corners=False
            )
            img = img_resized.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [600, 400, 3]

            if img.max() > 1:
                img = img / 255.0
            # 为每个子图指定位置和大小 [left, bottom, width, height]
            ax1 = fig.add_axes([0.05, 0.1, 0.4, 0.8])  # 左边子图
            ax2 = fig.add_axes([0.55, 0.1, 0.4, 0.8])  # 右边子图
            # ax3= fig.add_axes([0.55, 0.1, 0.4, 0.8])  # 右边子图
            # gs = GridSpec(1, 2, width_ratios=[1, 1])  # 两个子图宽度比例为1:1
            ax1.imshow(L,aspect='auto',cmap='hot')
            ax1.set_title('Original Image')
            ax1.axis('off')

            # 注意力图
            # im = ax2.imshow(att.cpu().numpy(), cmap='hot',aspect='auto')
            # ax2.set_title('Attention Map')
            # ax2.axis('off')
            # plt.colorbar(im,)
            # print(L.shape,att.shape)
            L_new=L*att+L
            im = ax2.imshow(L_new, cmap='hot',aspect='auto')
            ax2.set_title('Attention Map')
            ax2.axis('off')
            # plt.colorbar(im,)
            # 调整子图间距
            # plt.tight_layout()
            # plt.subplots_adjust(wspace=0.4)  # 水平间距，根据需要调整
            #
            # im = ax2.imshow(att.cpu().numpy(), cmap='hot', aspect='auto')
            plt.show()


        # plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_sample_{i}.png", dpi=300, bbox_inches='tight')

        plt.show()


def get_attention_with_hook(model, input_tensor):
    """使用钩子获取注意力图"""
    attention_maps = []

    def hook_fn(module, input, output):
        # 假设注意力图是第一个输出
        print(f"Hook output shape: {output.shape}")
        attention_maps.append(output.detach().cpu())

    # 注册钩子
    hook = model.lsa.l_gate.register_forward_hook(hook_fn)

    # 前向传播
    with torch.no_grad():
        output, delta_L,delta_Lc,delta_ab,L,L_new,ab,ab_new,L_att, ab_att,illum_L,reflect_L = model(input_tensor)

    # 移除钩子
    hook.remove()

    return output,attention_maps[0], delta_L,delta_Lc,delta_ab,L,L_new,ab,ab_new,L_att, ab_att,illum_L,reflect_L


def visualize_enhanced_results(enhanced_rgb, delta_L, original_image=None, save_path=None):
    """
    可视化增强结果和delta_L

    Args:
        enhanced_rgb: 增强后的RGB图像 [B, 3, H, W]
        delta_L: L通道的增量 [B, 1, H, W]
        original_image: 原始图像 [B, 3, H, W] (可选)
        save_path: 保存路径 (可选)
    """
    print(f"Enhanced RGB shape: {enhanced_rgb.shape}")
    print(f"Delta L shape: {delta_L.shape}")
    print(f"Delta L range - Min: {delta_L.min().item():.4f}, Max: {delta_L.max().item():.4f}")

    batch_size = enhanced_rgb.shape[0]

    for i in range(batch_size):
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))

        # 处理enhanced_rgb
        enhanced = enhanced_rgb[i]  # [3, H, W]
        enhanced_np = enhanced.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        # 归一化到[0, 1]
        if enhanced_np.max() > 1:
            enhanced_np = enhanced_np / 255.0
        enhanced_np = np.clip(enhanced_np, 0, 1)

        # 处理delta_L - 归一化到[0, 1]范围
        delta = delta_L[i]  # [1, H, W]
        delta_np = delta.squeeze(0).cpu().numpy()  # [H, W]

        # 归一化delta_L到[0, 1]范围
        delta_min = delta_np.min()
        delta_max = delta_np.max()
        if delta_max - delta_min > 1e-6:  # 避免除以0
            delta_np_normalized = (delta_np - delta_min) / (delta_max - delta_min)
        else:
            delta_np_normalized = delta_np * 0  # 如果所有值相同，设为0

        print(f"Sample {i} - Delta L range: [{delta_min:.4f}, {delta_max:.4f}]")

        # # 显示enhanced_rgb
        # axes[0].imshow(enhanced_np)
        # axes[0].set_title('Enhanced RGB')
        # axes[0].axis('off')
        #
        # # 显示归一化后的delta_L
        # im = axes[1].imshow(delta_np_normalized, cmap='coolwarm', vmin=0, vmax=1)
        # axes[1].set_title(f'Delta L (Normalized)\nRange: [{delta_min:.3f}, {delta_max:.3f}]')
        # axes[1].axis('off')
        # plt.colorbar(im, ax=axes[1])

        # 如果有原始图像，显示原始图像
        if original_image is not None:
            orig = original_image[i]  # [3, H, W]
            orig_np = orig.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

            if orig_np.max() > 1:
                orig_np = orig_np / 255.0
            orig_np = np.clip(orig_np, 0, 1)

            axes[2].imshow(orig_np)
            axes[2].set_title('Original Image')
            axes[2].axis('off')
        else:
            # 如果没有原始图像，显示delta_L的直方图
            axes[2].hist(delta_np.flatten(), bins=50, alpha=0.7, color='blue')
            axes[2].set_title('Delta L Distribution')
            axes[2].set_xlabel('Delta L Value')
            axes[2].set_ylabel('Frequency')

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_sample_{i}.png", dpi=300, bbox_inches='tight')

        plt.show()


def visualize_enhanced_results(enhanced_rgb, delta_L, original_image=None, save_path=None):
    """
    可视化增强结果和delta_L

    Args:
        enhanced_rgb: 增强后的RGB图像 [B, 3, H, W]
        delta_L: L通道的增量 [B, 1, H, W]
        original_image: 原始图像 [B, 3, H, W] (可选)
        save_path: 保存路径 (可选)
    """
    print(f"Enhanced RGB shape: {enhanced_rgb.shape}")
    print(f"Delta L shape: {delta_L.shape}")
    print(f"Delta L range - Min: {delta_L.min().item():.4f}, Max: {delta_L.max().item():.4f}")

    batch_size = enhanced_rgb.shape[0]

    for i in range(batch_size):
        # 根据是否有原始图像决定子图数量
        if original_image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 处理enhanced_rgb
        enhanced = enhanced_rgb[i]  # [3, H, W]
        enhanced_np = enhanced.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

        # 归一化到[0, 1]
        if enhanced_np.max() > 1:
            enhanced_np = enhanced_np / 255.0
        enhanced_np = np.clip(enhanced_np, 0, 1)

        # 处理delta_L - 归一化到[0, 1]范围
        delta = delta_L[i]  # [1, H, W]
        delta_np = delta.squeeze(0).cpu().numpy()  # [H, W]

        # 归一化delta_L到[0, 1]范围
        delta_min = delta_np.min()
        delta_max = delta_np.max()
        if delta_max - delta_min > 1e-6:  # 避免除以0
            delta_np_normalized = (delta_np - delta_min) / (delta_max - delta_min)
        else:
            delta_np_normalized = delta_np * 0  # 如果所有值相同，设为0

        print(f"Sample {i} - Delta L range: [{delta_min:.4f}, {delta_max:.4f}]")

        # 显示enhanced_rgb
        axes[0].imshow(enhanced_np)
        axes[0].set_title('Enhanced RGB')
        axes[0].axis('off')

        # 显示归一化后的delta_L
        im = axes[1].imshow(delta_np_normalized, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title(f'Delta L (Normalized)')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])

        # 如果有原始图像，显示原始图像
        if original_image is not None:
            orig = original_image[i]  # [3, H, W]
            orig_np = orig.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]

            if orig_np.max() > 1:
                orig_np = orig_np / 255.0
            orig_np = np.clip(orig_np, 0, 1)

            axes[2].imshow(orig_np)
            axes[2].set_title('Original Image')
            axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_sample_{i}.png", dpi=300, bbox_inches='tight')

        plt.show()


def visualize_delta_L_analysis(delta_L, save_path=None):
    """
    详细分析delta_L的分布 - 增强可视化效果

    Args:
        delta_L: L通道的增量 [B, 1, H, W]
        save_path: 保存路径 (可选)
    """
    print(f"Delta L shape: {delta_L.shape}")

    batch_size = delta_L.shape[0]

    for i in range(batch_size):
        delta = delta_L[i].squeeze(0).cpu().numpy()  # [H, W]

        # 计算统计信息
        delta_min = delta.min()
        delta_max = delta.max()
        delta_mean = delta.mean()
        delta_std = delta.std()
        delta_median = np.median(delta)

        print(f"Sample {i} - Delta L Statistics:")
        print(f"  Min: {delta_min:.4f}, Max: {delta_max:.4f}")
        print(f"  Mean: {delta_mean:.4f}, Std: {delta_std:.4f}, Median: {delta_median:.4f}")

        # 方法1: 标准归一化
        if delta_max - delta_min > 1e-6:
            delta_normalized = (delta - delta_min) / (delta_max - delta_min)
        else:
            delta_normalized = delta * 0

        # 方法2: 使用百分位数进行归一化，减少异常值影响
        p_low = np.percentile(delta, 2)  # 2%分位数
        p_high = np.percentile(delta, 98)  # 98%分位数
        if p_high - p_low > 1e-6:
            delta_percentile = (delta - p_low) / (p_high - p_low)
            delta_percentile = np.clip(delta_percentile, 0, 1)
        else:
            delta_percentile = delta_normalized

        # 方法3: 使用标准差进行归一化
        if delta_std > 1e-6:
            delta_std_norm = (delta - delta_mean) / (3 * delta_std) + 0.5  # 映射到[0,1]
            delta_std_norm = np.clip(delta_std_norm, 0, 1)
        else:
            delta_std_norm = delta_normalized

        # 方法4: 应用伽马校正增强对比度
        gamma = 0.5  # 小于1增强暗部细节
        delta_gamma = np.power(delta_normalized, gamma)

        # 创建多个子图比较不同方法
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 原始delta_L直方图
        axes[0, 0].hist(delta.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Original Delta L Distribution')
        axes[0, 0].set_xlabel('Delta L Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(delta_mean, color='red', linestyle='--', label=f'Mean: {delta_mean:.3f}')
        axes[0, 0].axvline(delta_median, color='green', linestyle='--', label=f'Median: {delta_median:.3f}')
        axes[0, 0].legend()

        # 标准归一化
        im1 = axes[0, 1].imshow(delta_normalized, cmap='hot', vmin=0, vmax=1)
        axes[0, 1].set_title('Standard Normalization')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])

        # 百分位数归一化
        im2 = axes[0, 2].imshow(delta_percentile, cmap='hot', vmin=0, vmax=1)
        axes[0, 2].set_title('Percentile Normalization\n(2%-98%)')
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2])

        # 标准差归一化
        im3 = axes[1, 0].imshow(delta_std_norm, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Std-based Normalization\n(Mean ± 3σ)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0])

        # 伽马校正
        im4 = axes[1, 1].imshow(delta_gamma, cmap='hot', vmin=0, vmax=1)
        axes[1, 1].set_title(f'Gamma Correction\n(γ={gamma})')
        axes[1, 1].axis('off')
        plt.colorbar(im4, ax=axes[1, 1])

        # 不同颜色映射比较
        im5 = axes[1, 2].imshow(delta_percentile, cmap='viridis', vmin=0, vmax=1)
        axes[1, 2].set_title('Viridis Colormap')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2])

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_delta_analysis_{i}.png", dpi=300, bbox_inches='tight')

        plt.show()


def visualize_delta_L_simple(delta_L, method='NONNE', cmap='viridis', gamma=0.5, save_path=None):
    """
    简化版的delta_L可视化，使用最佳方法

    Args:
        delta_L: L通道的增量 [B, 1, H, W]
        method: 归一化方法 ('standard', 'percentile', 'std', 'gamma')
        cmap: 颜色映射 ('hot', 'viridis', 'plasma', 'inferno', 'magma')
        gamma: 伽马校正值 (仅当method='gamma'时使用)
        save_path: 保存路径 (可选)
    """
    print(f"Delta L shape: {delta_L.shape}")

    batch_size = delta_L.shape[0]

    for i in range(batch_size):
        delta = delta_L[i].squeeze(0).cpu().numpy()  # [H, W]

        # 计算统计信息
        delta_min = delta.min()
        delta_max = delta.max()
        delta_mean = delta.mean()

        # 根据选择的方法进行归一化
        if method == 'standard':
            # 标准归一化
            if delta_max - delta_min > 1e-6:
                delta_norm = (delta - delta_min) / (delta_max - delta_min)
            else:
                delta_norm = delta * 0
            title = f'Standard Normalization\nRange: [{delta_min:.3f}, {delta_max:.3f}]'

        elif method == 'percentile':
            # 百分位数归一化
            p_low = np.percentile(delta, 2)
            p_high = np.percentile(delta, 98)
            if p_high - p_low > 1e-6:
                delta_norm = (delta - p_low) / (p_high - p_low)
                delta_norm = np.clip(delta_norm, 0, 1)
            else:
                delta_norm = (delta - delta_min) / (
                            delta_max - delta_min) if delta_max - delta_min > 1e-6 else delta * 0
            title = f'Percentile Normalization\n(2%-98%: [{p_low:.3f}, {p_high:.3f}])'

        elif method == 'std':
            # 标准差归一化
            delta_std = delta.std()
            if delta_std > 1e-6:
                delta_norm = (delta - delta_mean) / (3 * delta_std) + 0.5
                delta_norm = np.clip(delta_norm, 0, 1)
            else:
                delta_norm = (delta - delta_min) / (
                            delta_max - delta_min) if delta_max - delta_min > 1e-6 else delta * 0
            title = f'Std-based Normalization\n(Mean: {delta_mean:.3f}, Std: {delta_std:.3f})'

        elif method == 'gamma':
            # 伽马校正
            if delta_max - delta_min > 1e-6:
                delta_temp = (delta - delta_min) / (delta_max - delta_min)
                delta_norm = np.power(delta_temp, gamma)
            else:
                delta_norm = delta * 0
            title = f'Gamma Correction (γ={gamma})\nRange: [{delta_min:.3f}, {delta_max:.3f}]'

        else:
            delta_norm =delta
            # raise ValueError(f"Unknown method: {method}")

        # 创建可视化
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        im = ax.imshow(delta_norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_title("title")
        ax.axis('off')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()

        if save_path:
            plt.savefig(f"{save_path}_delta_simple_{i}.png", dpi=300, bbox_inches='tight')

        plt.show()

import os, math

def _to_numpy(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    return x

def _minmax01(arr, eps=1e-6):
    a = _to_numpy(arr)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < eps:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)

def _robust_zero_center(arr, p=2.0, eps=1e-6):
    """
    Robust zero-centered visualization:
    map negatives to <0.5, positives to >0.5, with percentile scaling to resist outliers.
    """
    a = _to_numpy(arr)
    pos = np.percentile(a, 100 - p)
    neg = np.percentile(a, p)
    m = max(abs(pos), abs(neg), eps)
    a01 = a / m * 0.5 + 0.5
    return np.clip(a01, 0.0, 1.0)


def _enhance_contrast(img01, low=2, high=98):
    """Percentile-based contrast stretch to [0,1]."""
    import numpy as _np
    a = img01
    p_low, p_high = _np.percentile(a, [low, high])
    # if p_high - p_low < 1e-9:
    #     return _np.zeros_like(a)
    a = (a - p_low) / (p_high - p_low)
    return _np.clip(a, 0.0, 1.0)

def _save_imshow(img01, title, save_dir, fname, add_colorbar=True):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        if hasattr(img01, 'ndim') and img01.ndim == 2:
            im = ax.imshow(img01, vmin=0.0, vmax=1.0, cmap='hot')
        else:
            im = ax.imshow(img01, vmin=0.0, vmax=1.0)
        ax.axis('off')
        # ax.set_title(title, pad=6)
        if add_colorbar and hasattr(img01, 'ndim') and img01.ndim == 2:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='3%', pad=0.04)
            cb = fig.colorbar(im, cax=cax)
        out = os.path.join(save_dir, fname)
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.02)
        return out
    finally:
        plt.close(fig)


def visualize_all(
    save_dir,
    output=None,           # [B,3,H,W], RGB [0,1]
    attention_map=None,    # [B,1,H,W] or [B,H,W], [0,1]
    delta_L=None,          # [B,1,H,W], unbounded
    delta_Lc=None,         # [B,1,H,W], in [-1,1]
    delta_ab=None,         # [B,2,H,W], unbounded
    L=None, L_new=None,    # [B,1,H,W], in [-1,1]
    ab=None, ab_new=None,  # [B,2,H,W], in [-1,1]
    L_att=None,            # [B,1,H,W], [0,1]
    ab_att=None,           # [B,2,H,W], [0,1]
    illum_L=None,
    reflect_L=None
):
    """
    Save normalized visualizations for all intermediate tensors.
    """
    def _ensure4d(x):
        if x is None: return None
        if x.dim() == 2: x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.size(0) not in (3,2):
                x = x.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        return x

    attention_map = _ensure4d(attention_map)
    delta_L      = _ensure4d(delta_L)
    delta_Lc     = _ensure4d(delta_Lc)
    L            = _ensure4d(L)
    L_new        = _ensure4d(L_new)
    L_att        = _ensure4d(L_att)
    # ab_att stays 4D if provided

    # Reference spatial size
    ref_hw = None
    if output is not None:
        ref_hw = output.shape[-2:]
    elif L is not None:
        ref_hw = L.shape[-2:]
    elif attention_map is not None:
        ref_hw = attention_map.shape[-2:]

    def _resize_like(x, hw):
        if x is None or hw is None: return x
        if x.shape[-2:] == hw: return x
        return F.interpolate(x, size=hw, mode="bilinear", align_corners=False)

    if ref_hw is not None:
        for name in ["attention_map","delta_L","delta_Lc","L","L_new","L_att"]:
            obj = locals()[name]
            if obj is not None:
                locals()[name] = _resize_like(obj, ref_hw)
        if delta_ab is not None and delta_ab.dim()==4:
            delta_ab = _resize_like(delta_ab, ref_hw)
        if ab is not None and ab.dim()==4:
            ab = _resize_like(ab, ref_hw)
        if ab_new is not None and ab_new.dim()==4:
            ab_new = _resize_like(ab_new, ref_hw)
        if ab_att is not None and ab_att.dim()==4:
            ab_att = _resize_like(ab_att, ref_hw)
        if output is not None:
            output = _resize_like(output, ref_hw)

    B = 1
    for x in [output, attention_map, delta_L, delta_Lc, L, L_new, ab, ab_new, delta_ab, L_att, ab_att]:
        if x is not None:
            B = int(x.shape[0]); break

    saved = []
    for i in range(B):
        tag = f"sample_{i:02d}"

        # output RGB
        if output is not None:
            rgb = _to_numpy(output[i].permute(1,2,0))
            rgb = np.clip(rgb, 0.0, 1.0)
            saved.append(_save_imshow(rgb, "output (RGB [0,1])", save_dir, f"{tag}_output.png", add_colorbar=False))


        # L_att (L-branch attention)
        if L_att is not None:
            print(L_att.max(),L_att.min())
            latt = _to_numpy(L_att[i,0])
            latt = np.clip(latt, 0.0, 1.0)
            latt = _robust_zero_center(latt)
            print(latt.max(), latt.min())
            saved.append(_save_imshow(latt, "L_att [0,1]", save_dir, f"{tag}_L_att.png"))

        # ab_att (ab-channel attention): per-channel + magnitude
        if ab_att is not None:
            abt = ab_att[i]
            a_att = _to_numpy(abt[0])
            b_att = _to_numpy(abt[1])
            a_att = _enhance_contrast(np.clip(a_att, 0.0, 1.0))
            b_att = _enhance_contrast(np.clip(b_att, 0.0, 1.0))
            mag_att = _enhance_contrast(np.sqrt(a_att*a_att + b_att*b_att) / np.sqrt(2.0))
            saved.append(_save_imshow(a_att, "ab_att.a [0,1]", save_dir, f"{tag}_ab_att_a.png"))
            saved.append(_save_imshow(b_att, "ab_att.b [0,1]", save_dir, f"{tag}_ab_att_b.png"))
            saved.append(_save_imshow(mag_att, "ab_att.magnitude", save_dir, f"{tag}_ab_att_mag.png"))

        # attention_map
        if attention_map is not None:
            att = _to_numpy(attention_map[i,0])
            att = np.clip(att, 0.0, 1.0)
            att = _enhance_contrast(att)
            saved.append(_save_imshow(att, "attention_map [0,1]", save_dir, f"{tag}_attention.png"))

        # delta_L
        if delta_L is not None:
            dL = _robust_zero_center(delta_L[i,0])
            saved.append(_save_imshow(dL, "delta_L (0.5 = 0)", save_dir, f"{tag}_delta_L.png"))

        # delta_Lc [-1,1]
        if delta_Lc is not None:
            dLc = _to_numpy(delta_Lc[i,0])
            dLc_01 = np.clip((dLc + 1.0) * 0.5, 0.0, 1.0)
            saved.append(_save_imshow(dLc_01, "delta_Lc ([-1,1]→[0,1])", save_dir, f"{tag}_delta_Lc.png"))

        # L, L_new [-1,1]
        if L is not None:
            Li = _to_numpy(L[i,0])
            Li_01 = np.clip((Li + 1.0) * 0.5, 0.0, 1.0)
            saved.append(_save_imshow(Li_01, "L ([-1,1]→[0,1])", save_dir, f"{tag}_L.png"))
        if L_new is not None:
            Ln = _to_numpy(L_new[i,0])
            Ln_01 = np.clip((Ln + 1.0) * 0.5, 0.0, 1.0)
            saved.append(_save_imshow(Ln_01, "L_new ([-1,1]→[0,1])", save_dir, f"{tag}_L_new.png"))
        if reflect_L is not None:
            Lii = _to_numpy(reflect_L[i,0])
            Li_01 = _enhance_contrast(Lii)
            saved.append(_save_imshow(Li_01, "reflect_L ([-1,1]→[0,1])", save_dir, f"{tag}_reflect_L.png"))
        if   illum_L is not None:
            Ln = _to_numpy(illum_L[i,0])
            Ln_01 = _enhance_contrast(Ln)
            saved.append(_save_imshow(Ln_01, "illum_L ([-1,1]→[0,1])", save_dir, f"{tag}_illum_L.png"))
        # ab / ab_new [-1,1] + chroma magnitude
        def _viz_ab(prefix, abab):
            a = _to_numpy(abab[i,0]); b = _to_numpy(abab[i,1])
            a01 = np.clip((a + 1.0) * 0.5, 0.0, 1.0)
            b01 = np.clip((b + 1.0) * 0.5, 0.0, 1.0)
            mag = np.sqrt(a*a + b*b) / math.sqrt(2.0)
            mag = _enhance_contrast(mag)
            saved.append(_save_imshow(a01, f"{prefix}.a ([-1,1]→[0,1])", save_dir, f"{tag}_{prefix}_a.png"))
            saved.append(_save_imshow(b01, f"{prefix}.b ([-1,1]→[0,1])", save_dir, f"{tag}_{prefix}_b.png"))
            saved.append(_save_imshow(mag, f"{prefix}.magnitude", save_dir, f"{tag}_{prefix}_mag.png"))

        if ab is not None:
            _viz_ab("ab", ab)
        if ab_new is not None:
            _viz_ab("ab_new", ab_new)

        # delta_ab: per-channel zero-centered + magnitude
        if delta_ab is not None:
            da = delta_ab[i,0]; db = delta_ab[i,1]
            da01 = _robust_zero_center(da)
            db01 = _robust_zero_center(db)
            damag = _to_numpy(torch.sqrt(da**2 + db**2))
            damag01 = _minmax01(damag)
            damag01 = _enhance_contrast(damag01)
            saved.append(_save_imshow(da01, "delta_ab.a (0.5=0)", save_dir, f"{tag}_delta_ab_a.png"))
            saved.append(_save_imshow(db01, "delta_ab.b (0.5=0)", save_dir, f"{tag}_delta_ab_b.png"))
            saved.append(_save_imshow(damag01, "delta_ab.magnitude", save_dir, f"{tag}_delta_ab_mag.png"))

    return saved

# 使用钩子方法的示例
if __name__ == '__main__':
    import torchvision.transforms as transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BCC_LabNet().cuda()
    model.load_state_dict(torch.load(r"VisDrone\BCC_LabNetbasev3new_psnr.pth", map_location=device), )
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载图像
    x = Image.open(r'data\VisDrone\train\low\9999993_00000_d_0000014_jpg.rf.ac9c8ff4222a80ea0997acdd9992eff6.jpg').convert('RGB')
    low_image = transform(x).cuda().unsqueeze(dim=0)

    print(f"Input image shape: {low_image.shape}")

    # 使用钩子获取注意力图
    output, attention_map, delta_L, delta_Lc, delta_ab, L, L_new, ab, ab_new,L_att, ab_att,illum_L,reflect_L  = get_attention_with_hook(model, low_image)
    saved_files = visualize_all(
        save_dir="./viz_outnnn",
        output=output,
        attention_map=attention_map,  # 注意力已在[0,1]
        delta_L=delta_L,  # 无界，自动稳健零中心
        delta_Lc=delta_Lc,  # [-1,1] → [0,1]
        delta_ab=delta_ab,  # 无界，零中心+强度
        L=L, L_new=L_new,  # [-1,1] → [0,1]
        ab=ab, ab_new=ab_new, # [-1,1] → [0,1]，并输出色度模
        L_att=L_att,
        ab_att=ab_att,
        illum_L=illum_L,reflect_L= reflect_L
    )

    print("Saved:", *saved_files, sep="\n")
    # Save unified visualizations to ./viz_out
    # try:
    #     _saved = visualize_all(
    #         save_dir="./viz_out",
    #         output=output,
    #         attention_map=attention_map,
    #         delta_L=delta_L,
    #         delta_Lc=delta_Lc,
    #         delta_ab=delta_ab,
    #         L=L, L_new=L_new,
    #         ab=ab, ab_new=ab_new
    #     )
    #     print("Visualization saved:")
    #     [print(f" - {pp}") for pp in _saved]
    # except Exception as e:
    #     print("visualize_all failed:", e)
    # print(delta_L.max(),delta_L.min())
    # if delta_L is not None:
    #     visualize_attention(attention_map,original_image=low_image.cpu(),L=L.squeeze(0).cpu())
    # else:
    #     print("No attention map captured!")

# =========================
# Unified visualization utils for tensors with different value ranges
# =========================