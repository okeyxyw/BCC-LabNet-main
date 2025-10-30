import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.color import rgb_to_lab, lab_to_rgb   # pip install kornia
class LSKernel(nn.Module):
    """大核可分离：k×k DW -> 1×k 条带 -> k×1 条带"""
    def __init__(self, channels, k):
        super().__init__()
        pad = k // 2
        self.dw = nn.Conv2d(
            channels, channels, k, 1, pad,
            groups=channels, bias=True
        )
        self.h_strip = nn.Conv2d(
            channels, channels, (1, k), 1, (0, pad),
            groups=channels, bias=True
        )
        self.v_strip = nn.Conv2d(
            channels, channels, (k, 1), 1, (pad, 0),
            groups=channels, bias=True
        )
    def forward(self, x):
        return self.dw(x) + self.h_strip(x) + self.v_strip(x)
class LSFusionBlock(nn.Module):
    def __init__(self, in_c, out_c, big_k=7, mid_k=5):
        super().__init__()

        # 保底：每个分支至少 1 通道，总和为 out_c
        if out_c < 3:
            big_c = mid_c = small_c = 1
            # 若 out_c == 2，合并两个分支
            if out_c == 2:
                big_c = 0
                mid_c = 1
                small_c = 1
        else:
            total = out_c
            big_c  = max(1, total // 6)
            mid_c  = max(1, total * 2 // 6)
            small_c = total - big_c - mid_c
            # 再次兜底
            if small_c < 1:
                mid_c -= 1
                small_c = 1

        # 小核分支（3×3）
        self.small_dw = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=True)
        self.small_pw = nn.Conv2d(in_c, small_c, 1, 1, 0, bias=True)

        # 中核分支（5×5）
        self.mid_kernel = LSKernel(in_c, mid_k) if mid_c > 0 else None
        self.mid_pw = nn.Conv2d(in_c, mid_c, 1, 1, 0, bias=True) if mid_c > 0 else None

        # 大核分支（7×7）
        self.big_kernel = LSKernel(in_c, big_k) if big_c > 0 else None
        self.big_pw = nn.Conv2d(in_c, big_c, 1, 1, 0, bias=True) if big_c > 0 else None

        # 融合
        self.fuse = nn.Conv2d(small_c+mid_c+big_c, out_c, 1, 1, 0, bias=True)
        self.act = nn.LeakyReLU()
        self.shortcut = nn.Identity() if in_c == out_c else \
                        nn.Conv2d(in_c, out_c, 1, 1, 0, bias=True)

    def forward(self, x):
        identity = self.shortcut(x)
        outs = []

        # 小核
        s = self.small_pw(self.small_dw(x))
        outs.append(s)

        # 中核
        if self.mid_kernel is not None:
            m = self.mid_pw(self.mid_kernel(x))
            outs.append(m)

        # 大核
        if self.big_kernel is not None:
            b = self.big_pw(self.big_kernel(x))
            outs.append(b)

        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        return self.act(out + identity)

class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5/YOLOv8
    Args:
        in_channels:  输入通道
        out_channels: 输出通道
        kernel_size:  MaxPool 的核大小，默认 5（YOLOv5 官方）
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(hidden_channels * 4, out_channels, 1, 1, bias=False)
        self.act   = nn.LeakyReLU()
        self.pool  = nn.MaxPool2d(kernel_size=kernel_size,
                                  stride=1,
                                  padding=kernel_size // 2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.act(self.conv2(torch.cat([x, y1, y2, y3], 1)))
class DynamicLSFusionBlock(nn.Module):
    def __init__(self, in_c, out_c, max_kernels=3, min_branch_channels=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.max_kernels = max_kernels
        self.min_branch_channels = min_branch_channels

        # 动态确定使用哪些核大小 (7x7, 5x5, 3x3)
        self.kernel_sizes = self._determine_kernel_sizes()

        # 动态分配通道数
        self.branch_channels = self._allocate_channels()

        # 创建分支
        self.branches = nn.ModuleList()
        for i, k in enumerate(self.kernel_sizes):
            if k == 3:
                # 小核分支
                dw = nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=True)
                pw = nn.Conv2d(in_c, self.branch_channels[i], 1, 1, 0, bias=True)
                self.branches.append(nn.Sequential(dw, pw))
            else:
                # 中大核分支
                kernel = LSKernel(in_c, k)
                pw = nn.Conv2d(in_c, self.branch_channels[i], 1, 1, 0, bias=True)
                self.branches.append(nn.Sequential(kernel, pw))

        # 融合层
        self.fuse = nn.Conv2d(sum(self.branch_channels), out_c, 1, 1, 0, bias=True)
        self.act = nn.LeakyReLU()
        self.shortcut = nn.Identity() if in_c == out_c else \
            nn.Conv2d(in_c, out_c, 1, 1, 0, bias=True)

    def _determine_kernel_sizes(self):
        """根据输出通道数决定使用哪些核大小"""
        if self.out_c < 3 * self.min_branch_channels:
            # 通道数太少，只使用小核
            return [3]

        # 默认使用3种核大小，但可以根据需要调整
        return [7, 5, 3][:min(self.max_kernels, 3)]

    def _allocate_channels(self):
        """动态分配各分支的通道数"""
        num_branches = len(self.kernel_sizes)
        total = self.out_c

        if num_branches == 1:
            return [total]

        # 基础分配：按比例分配
        base_allocation = [total // num_branches] * num_branches
        remaining = total - sum(base_allocation)

        # 将剩余通道分配给前面的分支
        for i in range(remaining):
            base_allocation[i] += 1

        # 确保每个分支至少有min_branch_channels
        adjusted = []
        remaining = total
        for alloc in base_allocation[:-1]:
            adjusted_alloc = max(self.min_branch_channels, alloc)
            adjusted.append(adjusted_alloc)
            remaining -= adjusted_alloc

        # 最后一个分支取剩余所有通道
        adjusted.append(max(self.min_branch_channels, remaining))

        return adjusted

    def forward(self, x):
        identity = self.shortcut(x)
        outs = []

        for branch in self.branches:
            outs.append(branch(x))

        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        return self.act(out + identity)
# --- 工具：标准化 Lab 到 [-1,1] 方便网络学习 ---
def lab_norm(lab):
    L, ab = lab[:, :1], lab[:, 1:]
    L  = (L - 50.) / 50.      # L ∈ [0,100] → [-1,1]
    ab = ab / 128.            # ab ∈ [-128,128] → [-1,1]
    return torch.cat([L, ab], dim=1)

def lab_denorm(lab_normed):
    L, ab = lab_normed[:, :1], lab_normed[:, 1:]
    L  = L * 50. + 50.
    ab = ab * 128.
    return torch.cat([L, ab], dim=1)
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.LeakyReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x
class UNetBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        n=max(1,ch_out//8)
        self.body1 = DynamicLSFusionBlock(ch_in, ch_out)
        self.gn=nn.GroupNorm(n,ch_out)
        self.shortcut = nn.Identity() if ch_in == ch_out else \
            nn.Conv2d(ch_in, ch_out, 1, bias=False)

    def forward(self, x):
        return self.gn(self.body1(x))+ self.shortcut(x)
class SplitGCBlock(nn.Module):
    """
    分块 Global Context Net.
    1. 按通道分组，每组内做 GC 建模；
    2. 用 Split-Attention 汇总各组结果。
    """
    def __init__(self, channels: int, groups: int = 1, ratio: int = 1):
        super().__init__()
        assert channels % groups == 0
        self.groups = groups
        self.channels_per_group = channels // groups

        # 每个 group 内部的 GC 建模：Squeeze -> Excitation
        self.gc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.channels_per_group, self.channels_per_group // ratio, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.channels_per_group // ratio, self.channels_per_group, 1),
            )
            for _ in range(groups)
        ])

        # Split-Attention：决定每组权重
        self.sa_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, groups, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        i=x
        # 按通道分组: [B*g, C/g, H, W]
        x = x.view(B * g, self.channels_per_group, H, W)

        # 各组分别做 GC
        gc_out = []
        for i, gc in enumerate(self.gc_blocks):
            offset = i * self.channels_per_group
            chunk = x[:, offset: offset + self.channels_per_group, :, :]
            gc_out.append(gc(chunk))  # [B*g, C/g, 1, 1]
        gc_out = torch.cat(gc_out, dim=1)  # [B*g, C, 1, 1]

        # Split-Attention
        sa_weight = self.sa_fc(x.view(B, C, H, W))         # [B, g, 1, 1]
        sa_weight = sa_weight.view(B, g, 1, 1, 1)           # [B, g, 1, 1, 1]

        gc_out = gc_out.view(B, g, self.channels_per_group, 1, 1)  # [B, g, C/g, 1, 1]
        gc_out = (gc_out * sa_weight).sum(dim=1)                   # [B, C/g, 1, 1] -> [B, C, 1, 1]

        return gc_out.sigmoid()
class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//r, 1, bias=False), nn.SiLU(inplace=True),
            nn.Conv2d(c//r, c, 1, bias=False)
        )
        self.spat = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # channel
        idin=x
        ca = torch.sigmoid(self.mlp(x))
        x = x * ca
        # spatial
        sa = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        sa = self.spat(sa)
        return x * sa +idin


class UpBlock(nn.Module):
    def __init__(self, ch_in, ch_skip, ch_out):
        super().__init__()
        self.ch_skip = ch_skip
        self.up = nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2)

        if ch_skip != 0:
            self.att_ = ch_in + ch_skip
            # 使用 DynamicLSFusionBlock 替代普通卷积
            self.conv = UNetBlock(ch_in + ch_skip, ch_out)
        else:
            self.att_ = ch_in
            self.conv = UNetBlock(ch_in, ch_out)

        #self.gc_block = CBAM(self.att_)
        self.act = nn.LeakyReLU()

    def forward(self, x, skip=None):
        x = self.up(x)
        # 处理尺寸不匹配
        if skip is not None and self.ch_skip != 0:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)

        #x = self.gc_block(x)
        return self.act(self.conv(x))

class NoiseSuppression(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 噪声抑制掩码
        )

    def forward(self, x):
        noise_mask = self.conv(x)  # 噪声掩码 (低光照区接近1)
        return x * noise_mask  # 抑制噪声
class SubNet(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        c = [ 16, 32, 128, 256]*ch_in
        self.enc = nn.ModuleList([
            UNetBlock(ch_in, c[0]),
            UNetBlock(c[0], c[1]),
            UNetBlock(c[1], c[2]),
            UNetBlock(c[2], c[3])
        ])

        self.pool = nn.AvgPool2d(2)
        self.sppf=SPPF(c[3],c[3])
        self.b=SplitGCBlock(c[3])
        self.dec = nn.ModuleList([
            UpBlock(c[3], 0, c[2]),      # 128 -> 64
            UpBlock(c[2], c[2], c[1]),   # 64  -> 32
            UpBlock(c[1], c[1], c[0]),   # 32  -> 16
            UpBlock(c[0], c[0], ch_out)  # 16  -> ch_out
        ])

    def forward(self, x):
        skips = []
        h = x
        for layer in self.enc:
            h = layer(h)
            skips.append(h)
            h = self.pool(h)
        h=self.sppf(h)
        h=self.b(h)*h
        for i, layer in enumerate(self.dec):
            skip = skips[-i - 1]
            h = layer(h, skip)
        return h

# --- Lab 注意力模块 ---
class LSA(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.l_spatial = nn.Sequential(
            nn.Conv2d(1, 8, 7, padding=3, groups=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 1, 7, padding=3, groups=1),
            nn.Sigmoid()
        )
        self.ab_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(2, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, lab):
        L, ab = lab[:, :1], lab[:, 1:]
        att_L = self.l_spatial(L)
        L = L * att_L
        att_ab = self.ab_channel(ab)
        att_ab = att_ab.expand_as(ab)
        ab = ab * att_ab
        return torch.cat([L, ab], dim=1)
class LSA_V2(nn.Module):
    """改进版 Lab Spatial Attention：深度可分离 + 残差 + 通道混洗"""
    def __init__(self, channels=3, groups=2):
        super().__init__()
        # assert channels % groups == 0
        self.groups = max(1,channels % groups)

        # L 分支：深度可分离卷积
        self.l_spatial = nn.Sequential(
            nn.Conv2d(1, 1, 7, padding=3, groups=1, bias=False),  # depthwise
            nn.Conv2d(1, 8, 1, bias=False),                       # pointwise
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(8, 8, 7, padding=3, groups=8, bias=False),  # depthwise
            nn.Conv2d(8, 1, 1),                                   # pointwise
            nn.Sigmoid()
        )

        # ab 分支：全局通道注意力 + 通道混洗
        self.ab_channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, 2 // groups, 1, groups=1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(2 // groups, 2, 1, groups=1, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.size()
        x = x.view(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous().view(b, c, h, w)
        return x

    def forward(self, lab):
        L, ab = lab[:, :1], lab[:, 1:]

        # L 空间注意力（带残差）
        att_L = self.l_spatial(L)
        L = L * att_L + L   # 残差

        # ab 通道注意力 + 混洗
        att_ab = self.ab_channel(ab)
        att_ab = att_ab.expand_as(ab)
        ab = ab * att_ab
        ab = self.channel_shuffle(ab, self.groups)

        return torch.cat([L, ab], dim=1)


class AdaptiveGamma(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
            #nn.Linear(1, 1)  # 输出gamma缩放因子
        )

    def forward(self, x):
        gamma_scale = self.net(x) * 2.5 + 0.5  # 映射到[0.5, 3.0]
        return x ** (1.0 / gamma_scale)


class LabRetinexDualNet(nn.Module):
    def __init__(self, ch_in=3, ch_out=3):
        super().__init__()
        assert ch_in == 3 and ch_out == 3, "Lab空间输入输出需为3通道"

        # ----------------------------
        # 亮度分支 (L通道处理)
        # ----------------------------
        self.l_illum_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # 光照图约束在[0,1]
        )

        self.l_reflect_enhance = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

        # ----------------------------
        # 色度分支 (ab通道处理)
        # ----------------------------
        self.ab_enhance = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围[-1,1]对应ab通道
        )

        # 残差连接
        self.shortcut = nn.Identity()

    def forward(self, lab_img):
        """
        输入: Lab图像 [B,3,H,W] (L范围[0,100], ab范围[-128,127])
        输出: 增强后的Lab图像 [B,3,H,W]
        """
        # 标准化
        L = lab_img[:, :1, :, :] / 100.0  # L -> [0,1]
        ab = lab_img[:, 1:, :, :] / 128.0  # ab -> [-1,1]

        # ===== 亮度通道处理 =====
        # 1. Retinex分解
        illum_L = self.l_illum_net(L)  # 光照分量 [0,1]
        reflect_L = L / (illum_L + 1e-6)  # 反射分量

        # 2. 反射增强
        enhanced_L = self.l_reflect_enhance(reflect_L) * illum_L  # 重建

        # ===== 色度通道处理 =====
        enhanced_ab = self.ab_enhance(ab)  # 直接增强色度

        # ===== 合并结果 =====
        enhanced_L = enhanced_L * 100.0  # 恢复[0,100]范围
        enhanced_ab = enhanced_ab * 128.0  # 恢复[-128,127]范围
        output = torch.cat([enhanced_L, enhanced_ab], dim=1)

        return output + self.shortcut(lab_img)  # 残差连接
# ---------- 交叉注意力 ----------
class LabCrossAttn(nn.Module):
    def __init__(self, spatial_reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # L → ab
        self.q_L  = nn.Conv2d(1, 1, 1, bias=False)
        self.k_ab = nn.Conv2d(2, 1, 1, bias=False)
        self.v_ab = nn.Conv2d(2, 2, 1, bias=False)

        # ab → L
        self.q_ab = nn.Conv2d(2, 1, 1, bias=False)
        self.k_L  = nn.Conv2d(1, 1, 1, bias=False)
        self.v_L  = nn.Conv2d(1, 1, 1, bias=False)

        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _cross(q, k, v):
        # 确保所有输入都是 [B, C, 1, 1] 形状
        q = q.squeeze(-1).squeeze(-1)  # [B, C_q]
        k = k.squeeze(-1).squeeze(-1)  # [B, C_k]
        v = v.squeeze(-1).squeeze(-1)  # [B, C_v]

        # 计算注意力权重 [B, 1, 1]
        attn = torch.bmm(q.unsqueeze(1), k.unsqueeze(2))  # [B, 1, 1]
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到值 [B, C_v]
        out = attn.squeeze(1) * v  # [B, C_v]

        # 重塑为 [B, C_v, 1, 1]
        return out.unsqueeze(-1).unsqueeze(-1)
    # @staticmethod
    # def _cross(q, k, v):
    #     q, k, v = [x.squeeze(-1).squeeze(-1) for x in (q, k, v)]
    #     attn = torch.bmm(q.unsqueeze(1), k.unsqueeze(2))  # [B, 1, 1]
    #     attn = attn.softmax(dim=-1)
    #     out = attn * v  # [B, C]
    #     return out.view(v.size(0), v.size(1), 1, 1)  # [B, C, 1, 1]

    def forward(self, L, ab):
        L_g  = self.pool(L)
        ab_g = self.pool(ab)

        # L → ab
        ab_res = self._cross(self.q_L(L_g), self.k_ab(ab_g), self.v_ab(ab_g))
        ab_out = ab + ab_res.expand_as(ab)

        # ab → L
        L_res  = self._cross(self.q_ab(ab_g), self.k_L(L_g), self.v_L(L_g))
        L_out  = L + L_res.expand_as(L)

        return L_out, ab_out


# ---------- LSA_V2R 融合 ----------
class LSA_V2R_X(nn.Module):
    """
    LSA_V2R + LabCrossAttn
    输入: Lab [B,3,H,W]（已归一化）
    输出: 同 shape
    """
    def __init__(self, channels=3, groups=2):
        super().__init__()

        # ---- L 空间注意力（多尺度 DWConv） ----
        self.l_dw3  = nn.Conv2d(1, 4, 3, padding=1, groups=1, bias=False)
        self.l_dw5  = nn.Conv2d(1, 4, 5, padding=2, groups=1, bias=False)
        self.l_fuse = nn.Conv2d(8, 1, 1)
        self.l_gate = nn.Sigmoid()

        # ---- ab 通道注意力（轻量 SE） ----
        mid = max(1, 2 // 2)
        self.ab_gap = nn.AdaptiveAvgPool2d(1)
        self.ab_fc  = nn.Sequential(
            nn.Conv2d(2, mid, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid, 2, 1, bias=False),
            nn.Sigmoid()
        )
        self.global_res = nn.Parameter(torch.tensor(0.2))

        # ---- 交叉注意力 ----
        self.cross = LabCrossAttn()

    def forward(self, lab):
        L, ab = lab[:, :1], lab[:, 1:]

        # 1) 先各自做局部注意力
        f3 = self.l_dw3(L)
        f5 = self.l_dw5(L)
        att_L = self.l_gate(self.l_fuse(torch.cat([f3, f5], dim=1)))
        L = L * att_L + L

        att_ab = self.ab_fc(self.ab_gap(ab)).expand_as(ab)
        ab = ab * att_ab + ab * self.global_res

        # 2) 再做 L↔ab 交叉注意力
        L, ab = self.cross(L, ab)

        return torch.cat([L, ab], dim=1)
# ======= Drop-in: LSA_V3（替换 LSA/LSA_V2 调用）=======
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, c_in, c_out, k=7, s=1, p=None):
        super().__init__()
        if p is None: p = k // 2
        self.dw = nn.Conv2d(c_in, c_in, k, s, p, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = self.dw(x); x = self.pw(x); x = self.bn(x); return self.act(x)

def channel_shuffle(x, groups=2):
    b, c, h, w = x.shape
    g = max(1, min(groups, c))  # 兜底
    x = x.view(b, g, c // g, h, w).transpose(1, 2).contiguous()
    return x.view(b, c, h, w)

class SE2(nn.Module):
    """SE for 2ch ab"""
    def __init__(self, r=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, max(1,2//r), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(1,2//r), 2, 1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.net(x)
        return x * w

class BoxBlur(nn.Module):
    """引导低频：帮助L分支更稳"""
    def __init__(self, k=5):
        super().__init__()
        self.pool = nn.AvgPool2d(k, stride=1, padding=k//2, count_include_pad=False)
    def forward(self, x):
        return self.pool(x)

class LSA_V3(nn.Module):
    """
    输入/输出: [B,3,H,W] 的 Lab，L∈[0,100], ab∈[-128,128]（或已做你网内的归一化流）
    """
    def __init__(self, groups=2):
        super().__init__()
        # L 分支：深度可分离卷积 + 残差 + 低频引导
        self.l_dw = DepthwiseSeparableConv(1, 1, k=7)
        self.l_blur = BoxBlur(k=5)

        # ab 分支：SE 注意力 + 通道混洗 + 逐点细化
        self.ab_se = SE2(r=2)
        self.ab_pw = nn.Sequential(
            nn.Conv2d(2, 2, 1, bias=False),
            nn.InstanceNorm2d(2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.groups = groups

    def forward(self, lab):
        # 拆分
        L  = lab[:, :1]
        ab = lab[:, 1:]

        # L: 引导低频 + 空间注意 + 残差
        l_low = self.l_blur(L)
        l_att = torch.sigmoid(self.l_dw(L))
        L_out = L * l_att + 0.1 * l_low  # 轻残差引导，避免过度增强

        # ab: SE 通道注意 + 混洗 + 逐点细化（防色偏/提升饱和度稳定）
        ab = self.ab_se(ab)
        ab = channel_shuffle(ab, groups=self.groups)
        ab_out = self.ab_pw(ab)

        return torch.cat([L_out, ab_out], dim=1)

# --- 完整网络：BCC-LabNet ---
class BCC_LabNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.brightness_net = SubNet(1, 1)  # 输入1通道(L)，输出1通道(ΔL)
        self.contrast_net = SubNet(1, 1)  # 同上
        self.color_net = SubNet(2, 2)  # 输入2通道(ab)，输出2通道(Δab)
        self.lsa = LSA_V2R_X()
        self.AdaptiveGamma=AdaptiveGamma()
        self.Retinex=LabRetinexDualNet()
        self.lsah = LSA_V3()
    def forward(self, x_rgb):
        # x_rgb = self.lowlight_stabilizer(x_rgb)
        x_rgb=self.AdaptiveGamma(x_rgb)
        lab = rgb_to_lab(x_rgb)
        lab = lab_norm(lab)
        lab = self.lsa(lab)
        L, ab = lab[:, :1], lab[:, 1:]
        delta_L = self.brightness_net(L)
        delta_Lc = torch.tanh(self.contrast_net(L))
        delta_ab = self.color_net(ab)
        ab_new = ab + delta_ab
        L_new = (L + delta_L) * (1 + delta_Lc)
        lab_enh = torch.cat([L_new, ab_new], dim=1)
        lab_enh = self.lsah(lab_enh)
        lab_enh = lab_denorm(lab_enh)
        lab_enh = self.Retinex(lab_enh)
        rgb_enh = lab_to_rgb(lab_enh, clip=True)

        return rgb_enh.clamp_(0, 1)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- 测试 ---
if __name__ == '__main__':
    model = BCC_LabNet()
    x = torch.randn(2, 3,256,256)  # 任意尺寸（需为16的倍数）
    y = model(x)
    print(y.shape)  # torch.Size([1, 3, 600, 400])
    print("Total trainable parameters:", count_parameters(model))