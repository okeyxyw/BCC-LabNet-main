import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim
import torchvision.transforms as T
def mean_loss(y_true, y_pred):
    return torch.abs(torch.mean(y_true) - torch.mean(y_pred))
class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features[:16]  # Until block3_conv3
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))

def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 45.0 - torch.mean(psnr)

def smooth_l1_loss(y_true, y_pred):
    return F.mse_loss(y_true, y_pred)

def multiscale_ssim_loss(y_true, y_pred, max_val=1.0, power_factors=[0.5, 0.5]):
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)

def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)
    
    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))
    return hist_distance

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss_model = VGGPerceptualLoss(device)
        self.alpha1 = 1  # smooth_l1 (降低，避免过度平滑)
        self.alpha2 = 0.1  # perceptual (提高，保持细节)
        self.alpha3 = 0.05  # histogram (降低)
        self.alpha4 = 0.6  # ms-ssim (提高，保持结构)
        self.alpha5 = 0.2  # psnr (降低)
        self.alpha6 = 0.4  # color (保持)
        self.alpha7 = 0.4  # mean loss (适中)
    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        # ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss_model(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_loss(y_true, y_pred)
        mean_l = mean_loss(y_true, y_pred)
        total_loss = (
                self.alpha1 * smooth_l1_l +
                self.alpha2 * perc_l +
                self.alpha3 * hist_l +
                self.alpha4 * ms_ssim_l+
                self.alpha5 * psnr_l +
                self.alpha6 * color_l +
                self.alpha7 * mean_l  # 新增
        )

        return torch.mean(total_loss)
