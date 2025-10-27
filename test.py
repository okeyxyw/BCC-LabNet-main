import pyiqa
import torch
import torch.nn.functional as F
import math
from torchmetrics.functional import structural_similarity_index_measure
from torch.utils.data import DataLoader
from dataloader import create_dataloaders,LowOnlyDataset
import os
import numpy as np
from torchvision.utils import save_image
from base.BCCCOPYv3 import BCC_LabNet
import cv2
import pyiqa
def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=False):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def ssim(prediction, target):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2
    img1 = prediction.astype(np.float64)
    img2 = target.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=False):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    # 转换为 numpy，并缩放到 [0, 255]
    img1_np = (img1.squeeze(0).mean(dim=0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
    img2_np = (img2.squeeze(0).mean(dim=0) * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

    # 计算 SSIM
    ssim_val = ssim(img1_np, img2_np)
    return ssim_val
def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for idx, (low, high,name) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)
            # print(result_dir)
            # Save the output image
            save_image(output, os.path.join(result_dir, name[0]))

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim



def niqe_validate(model, dataloader, device, result_dir):
    model.eval()
    total_niqe = 0
    total_uicm = 0
    with torch.no_grad():
        for idx, (low, _) in enumerate(dataloader):
            low= low.to(device)
            # output = model(low)
            output = torch.clamp(low, 0, 1)
            # print(result_dir)
            # Save the output image
            save_image(output, os.path.join(result_dir, f'result_{idx}.png'))

            # Calculate niqe
            niqe = niqe_bchw(output)
            total_niqe += niqe



    avg_niqe = total_niqe / len(dataloader)
    return avg_niqe

def niqe_bchw(x: torch.Tensor,
             return_per_image: bool = False):
    """
    直接用 pyiqa 计算 NIQE。
    约定：
      - x: B×C×H×W，C∈{1,3}，数值范围已在 [0,1]（不做任何归一化/缩放）
      - device 不传则用 x.device
    返回：
      - return_per_image=False -> float（batch 平均）
      - return_per_image=True  -> (B,) 的 CPU 张量（逐图分数）
    """
    dev = x.device
    metric = pyiqa.create_metric('niqe', device=dev, as_loss=False)
    scores = metric(x.to(dev, dtype=torch.float32))  # (B,)

    return scores.detach().cpu().mean() if return_per_image else float(scores.mean().item())


def main():
    # Paths and device setup
    # train_low = './data/LOLv2/Synthetic/Train/Low'
    # train_high = r'./data/LOLv2/Synthetic/Train/Normal'
    # test_low = './data/LOLv2/Synthetic/Test/Low'
    # test_high = r'./data/LOLv2/Synthetic/Test/Normal'
    # weights_path = r'F:\BCC-Net-main\lolv2\BCC_LabNetbasev2sy_psnr.pth'
    # test_low = './data/LOLv2/Real_captured/Test/Low'
    # test_high = r'./data/LOLv2/Real_captured/Test/Normal'
    # weights_path = r'F:\BCC-Net-main\lolv2\BCC_LabNetbaseReal_psnr.pth'
    # train_low = 'data/LOLv1/Train/input'
    # train_high = 'data/LOLv1/Train/target'strict=False
    test_low =  'data/LOLv1/Test/input'
    test_high =  'data/LOLv1/Test/target'
    # train_low = 'F:\data\LoLI-Street Dataset\Train\input'
    # train_high = 'F:\data\LoLI-Street Dataset\Train\high'
    # test_low =  r'F:\data\LoLI-Street\val\input'
    # test_high =  r'F:\data\LoLI-Street\val\high'
    # test_low = r'F:\BCC_LabNet_Unpaired_GAN_Full\datasets\full'
    # test_high = r'data\VisDrone\test\higt'
    test_low = r'F:\data\LoLI-Street\Val\low'
    test_high = r'F:\data\LoLI-Street\Val\high'
    weights_path = r"lolv1\BCC_LabNetbasev3new_psnr.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test_ds = LowOnlyDataset(test_low,256)
    # test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # dataset_name = test_low.split('/')[1]
    result_dir = os.path.join('LoLI-Street')
    os.makedirs(result_dir, exist_ok=True)

    _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=1)
    print(f'Test loader: {len(test_loader)}')

    model = BCC_LabNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device),)
    print(f'Model loaded from {weights_path}')

    avg_psnr, avg_ssim= validate(model, test_loader, device, result_dir)
    print(avg_psnr, avg_ssim)


if __name__ == '__main__':
    main()
