import math
import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True, brightness_augmentation=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training
        self.brightness_augmentation = brightness_augmentation

        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])
        image_name = self.high_images[idx]
        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        # 亮度增强 - 仅对训练集且启用亮度增强时应用
        if self.training and self.brightness_augmentation:
            brightness_factor = random.uniform(0.5, 1.2)  # 亮度调整因子
            low_image = transforms.functional.adjust_brightness(low_image, brightness_factor)

        if self.training and self.crop_size:
            # 随机裁剪
            # if random.random()<0.4:
            i, j, h, w = transforms.RandomCrop.get_params(low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

            aug = random.randint(0, 8)
            # Data Augmentations
            if aug == 1:
                low_image = low_image.flip(1)
                high_image = high_image.flip(1)
            elif aug == 2:
                low_image = low_image.flip(2)
                high_image = high_image.flip(2)
            elif aug == 3:
                low_image = low_image.flip(1)
                high_image = high_image.flip(1)
                low_image = low_image.flip(2)
                high_image = high_image.flip(2)
            elif aug == 4:
                low_image = low_image.flip(2)
                high_image = high_image.flip(2)
                low_image = low_image.flip(1)
                high_image = high_image.flip(1)
            elif aug == 5:
                low_image = low_image.flip(1)
                high_image = high_image.flip(2)
                low_image = low_image.flip(1)
                high_image = high_image.flip(2)
            elif aug == 6:
                low_image = low_image.flip(2)
                high_image = high_image.flip(1)
                low_image = low_image.flip(2)
                high_image = high_image.flip(1)

        return low_image, high_image, image_name




class SIDDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])

        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')
        # w, h = low_image.size
        # new_h = math.ceil(h / 16) * 16
        # new_w = math.ceil(w / 16) * 16
        # low_image = low_image.resize((new_w, new_h), Image.BILINEAR)
        # high_image = high_image.resize((new_w, new_h), Image.BILINEAR)

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)
        # if self.training :
        #     gamma = np.random.uniform(0.5, 2.5)
        #     low_image = gamma_rgb(low_image, gamma)


        if self.training and self.crop_size:
            # if random.random()<0.8:
            i, j, h, w = transforms.RandomCrop.get_params(low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

            aug = random.randint(0, 8)
            # Data Augmentations
            if aug == 1:
                low_image = low_image.flip(1)
                high_image = high_image.flip(1)
            elif aug == 2:
                low_image = low_image.flip(2)
                high_image = high_image.flip(2)
            elif aug == 3:
                low_image = torch.rot90(low_image, dims=(1, 2))
                high_image = torch.rot90(high_image, dims=(1, 2))
            elif aug == 4:
                low_image = torch.rot90(low_image, dims=(1, 2), k=2)
                high_image = torch.rot90(high_image, dims=(1, 2), k=2)
            elif aug == 5:
                low_image = torch.rot90(low_image, dims=(1, 2), k=3)
                high_image = torch.rot90(high_image, dims=(1, 2), k=3)
            elif aug == 6:
                low_image = torch.rot90(low_image.flip(1), dims=(1, 2))
                high_image = torch.rot90(high_image.flip(1), dims=(1, 2))
            elif aug == 7:
                low_image = torch.rot90(low_image.flip(2), dims=(1, 2))
                high_image = torch.rot90(high_image.flip(2), dims=(1, 2))

        # print(low_image.max(),low_image.min())
        return low_image, high_image
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
import os, glob, random
def list_images(folder: str) -> list:
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(os.path.join(folder, f"*{ext}"))
    files.sort()
    return files
class LowOnlyDataset(Dataset):
    """
    Test/val dataset: only low-light images are needed for evaluation.
    """
    def __init__(self, low_dir: str, image_size: int = 256):
        self.low_paths = list_images(low_dir)
        if len(self.low_paths) == 0:
            raise FileNotFoundError("Test low folder is empty.")
        self.crop_size = image_size
        self.to_tensor = transforms.Compose([
  #          transforms.Resize((image_size, image_size)) if image_size and image_size>0 else transforms.Lambda(lambda x: x),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx: int):
        p = self.low_paths[idx]
        low_img = Image.open(p).convert("RGB")
        # i, j, h, w = transforms.RandomCrop.get_params(low_img, output_size=(self.crop_size, self.crop_size))
        # low_img = transforms.functional.crop(low_img, i, j, h, w)
        return self.to_tensor(low_img), os.path.basename(p)



def createlow_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = None
    test_loader = None

    if train_low and train_high:

            # print(400)
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # else:
        #     print('all')
        #     train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=False, training=True)
        #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if test_low and test_high:
        test_ds = LowOnlyDataset(test_low,256)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader

def create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_loader = None
    test_loader = None

    if train_low and train_high:

            # print(400)
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        # else:
        #     print('all')
        #     train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=False, training=True)
        #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    if test_low and test_high:
        test_dataset = PairedDataset(test_low, test_high, transform=transform, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    return train_loader, test_loader
