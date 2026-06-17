from time import time
import torch
from losses import CombinedLoss
from dataloader import create_dataloaders
import numpy as np
from base.BCCCOPYv5 import  BCC_LabNet
import cv2
import torch.nn.functional as F
import torch
torch.autograd.set_detect_anomaly(True)
# def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=False):
#     """
#     Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
#
#     Args:
#         img1 (torch.Tensor): First image (BxCxHxW)
#         img2 (torch.Tensor): Second image (BxCxHxW)
#         max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.
#
#     Returns:
#         float: The PSNR value.
#     """
#     img1_gray = img1.mean(axis=1)
#     img2_gray = img2.mean(axis=1)
#     if gt_mean:
#         mean_restored = img1_gray.mean()
#         mean_target = img2_gray.mean()
#         img1_gray = torch.clamp(img1_gray * (mean_target / mean_restored), 0, 1)
#
#     mse = F.mse_loss(img1_gray, img2_gray, reduction='mean')
#     if mse == 0:
#         return float('inf')
#     psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
#     return psnr.item()


# def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
#     """
#     Calculate SSIM (Structural Similarity Index) between two images.
#
#     Args:
#         img1 (torch.Tensor): First image (BxCxHxW)
#         img2 (torch.Tensor): Second image (BxCxHxW)
#         max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.
#
#     Returns:
#         float: The SSIM value.
#     """
#     if gt_mean:
#         img1_gray = img1.mean(axis=1, keepdim=True)
#         img2_gray = img2.mean(axis=1, keepdim=True)
#
#         mean_restored = img1_gray.mean()
#         mean_target = img2_gray.mean()
#         img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
#
#     ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
#     return ssim_val.item()

#
# def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=False):
#     """
#     计算两幅图像的 SSIM。
#     当 gt_mean=True 时，先用灰度均值把 img1 的亮度拉到与 img2 一致，再算 SSIM。
#     Args:
#         img1, img2 (torch.Tensor): 形状均为 (B,C,H,W)，取值范围 [0, max_pixel_value]
#         max_pixel_value (float): 图像最大像素值
#         gt_mean (bool): 是否做亮度对齐
#     Returns:
#         float: SSIM 值
#     """ # 转灰度：沿着 C 维求平均，保留 B 与 H、W 维
#     gray1 = img1.mean(dim=1, keepdim=True)      # (B,1,H,W)
#     gray2 = img2.mean(dim=1, keepdim=True)
#     if gt_mean:
#
#
#         # 每个样本单独算均值，避免 batch 内互相干扰
#         mean1 = gray1.view(gray1.size(0), -1).mean(dim=1)  # (B,)
#         mean2 = gray2.view(gray2.size(0), -1).mean(dim=1)
#
#         # 防止除 0
#         scale = torch.where(mean1 > 1e-6, mean2 / mean1, torch.ones_like(mean1))
#         scale = scale.view(-1, 1, 1, 1)  # (B,1,1,1) 便于广播
#
#         img1 = torch.clamp(img1 * scale, 0, max_pixel_value)
#
#     # 计算 SSIM，torchmetrics 会自动在 (C,H,W) 维度上求平均
#     ssim_val = structural_similarity_index_measure(
#         gray1, gray2, data_range=max_pixel_value
#     )
#     return ssim_val.item()
# def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
#     """
#     Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
#
#     Args:
#         img1 (torch.Tensor): First image (BxCxHxW)
#         img2 (torch.Tensor): Second image (BxCxHxW)
#         max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.
#
#     Returns:
#         float: The PSNR value.
#     """
#     if gt_mean:
#         img1_gray = img1.mean(axis=1)
#         img2_gray = img2.mean(axis=1)
#
#         mean_restored = img1_gray.mean()
#         mean_target = img2_gray.mean()
#         img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
#
#     mse = F.mse_loss(img1, img2, reduction='mean')
#     if mse == 0:
#         return float('inf')
#     psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
#     return psnr.item()
#
# def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
#     """
#     Calculate SSIM (Structural Similarity Index) between two images.
#
#     Args:
#         img1 (torch.Tensor): First image (BxCxHxW)
#         img2 (torch.Tensor): Second image (BxCxHxW)
#         max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.
#
#     Returns:
#         float: The SSIM value.
#     """
#     if gt_mean:
#         img1_gray = img1.mean(axis=1, keepdim=True)
#         img2_gray = img2.mean(axis=1, keepdim=True)
#
#         mean_restored = img1_gray.mean()
#         mean_target = img2_gray.mean()
#         img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)
#
#     ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
#     return ssim_val.item()
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
def validate(model, dataloader, device,):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for idx, (low, high,_) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)
            # print(result_dir)
            # Save the output image
          #  save_image(output, os.path.join(result_dir, f'result_{idx}.png'))

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
    # G:\VisDrone
    # train_low = r'F:\HVI-CIDNet-master\HVI-CIDNet-master\datasets\VisDrone\train\input'
    # train_high = r'F:\HVI-CIDNet-master\HVI-CIDNet-master\datasets\VisDrone\train\target'
    # test_low =  r'F:\HVI-CIDNet-master\HVI-CIDNet-master\datasets\VisDrone\test\input'
    # test_high =  r'F:\HVI-CIDNet-master\HVI-CIDNet-master\datasets\VisDrone\test\target'
    # Hyperparameters
    # train_low = 'data/LOLv1/Train/input'
    # train_high = 'data/LOLv1/Train/target'
    # test_low =  'data/LOLv1/Test/input'
    # test_high =  'data/LOLv1/Test/target'
    train_low = 'data/LOL/Train/input'
    train_high = 'data/LOL/Train/target'
    test_low =  'data/LOL/Test/input'
    test_high =  'data/LOL/Test/target'
    # train_low = 'F:\data\LoLI-Street Dataset\Train\input'
    # train_high = 'F:\data\LoLI-Street Dataset\Train\high'
    # test_low =  'F:\data\LoLI-Street Dataset\Val\input'
    # test_high =  'F:\data\LoLI-Street Dataset\Val\high'
    # train_low = 'data/LoLI-Street/Train/input'
    # train_high = 'data/LoLI-Street/Train/high'
    # test_low =  'data/LoLI-Street/Train/Val/input'
    # test_high =  'data/LoLI-Street/Val/high'
    # train_low = './data/LOLv2/Real_captured/Train/Low'
    # train_high = r'./data/LOLv2/Real_captured/Train/Normal'
    # test_low = './data/LOLv2/Real_captured/Test/Low'
    # test_high = r'./data/LOLv2/Real_captured/Test/Normal'
    # train_low = r'F:\data\sid\data\train\input'
    # train_high = r'F:\data\sid\data\train\target'
    # test_low = r'F:\data\sid\data\test\input'
    # test_high = r'F:\data\sid\data\test\target'
    # train_low = './data/LOLv2/Synthetic/Train/Low'
    # train_high = r'./data/LOLv2/Synthetic/Train/Normal'
    # test_low = './data/LOLv2/Synthetic/Test/Low'
    # test_high = r'./data/LOLv2/Synthetic/Test/Normal'
    # test_low = r'data/LSRW/Eval/Huawei/input'
    # test_high = r'data/LSRW/Eval/Huawei/high'
    # train_low = 'data/LSRW/Training/Huawei/input'
    # train_high = r'data/LSRW/Training/Huawei/high'
    # # train_low = r'F:\data\sid\data\train\input'
    # train_high = r'F:\data\sid\data\train\target'
    # test_low = r'F:\data\sid\data\test\input'
    # test_high = r'F:\data\sid\data\test\target'

    learning_rate = 1e-4
    num_epochs = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # Model, loss, optimizer, and scheduler
    model = BCC_LabNet().to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    criterion = CombinedLoss(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  betas=(0.9, 0.999),
                                  eps=1e-6,
                                  weight_decay=1e-4)  # ①  decay 放这里！
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1  # warmup比例
    )
    # scaler = torch.cuda.amp.GradScaler()
    model.load_state_dict(torch.load(r"lolv1/BCC_LabNetbasev3new_psnr.pth",map_location="cuda:0"),strict=False)#,
    # model.eval()
    best_psnr =23.901509475708007
    best_ssim =0.8874852457112326
    L1_weight=1.0
    # L1_loss = L1Loss(loss_weight=L1_weight, reduction='mean').cuda()
    print('Training started.')
    for epoch in range(num_epochs):
        t0 = time()
        model.train()
        train_loss = 0.0
        losssum=[]
        HVI_weight=1.0
        P_weight=0.01
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets,_ = batch
            inputs, gt_rgb = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            output_rgb= model(inputs)
            # output_hvi = model.HVIT(output_rgb),loss_hist,gt_rgb
            # gt_hvi = model.HVIT(gt_rgb)
            # loss_hvi = L1_loss(output_hvi, gt_hvi)
            loss = criterion(output_rgb, gt_rgb)
            # loss =  HVI_weight * loss_hvi + lab_loss
            # print(outputs.max(),outputs.min())
            losssum.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            train_loss += loss.item()
        avg_psnr, avg_ssim = validate(model, test_loader, device)
        # print(avg_psnr)
        t1 = time()
        print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, time: {(t1 - t0):.2f}, ')
        scheduler.step()

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), 'lolv1/BCC_LabNetbasev5newn_psnr.pth')
            print(f'Saving model with PSNR: {best_psnr:.6f}')
        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            torch.save(model.state_dict(), 'lolv1/BCC_LabNetbasev5newn_ssim.pth')
            print(f'Saving model with SSIM: {best_ssim:.6f}')
if __name__ == '__main__':
    main()
