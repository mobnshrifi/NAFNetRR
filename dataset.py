"""
Dataset Loader for Image Restoration
لودر داده برای بازسازی تصویر
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path


class ImageRestorationDataset(Dataset):
    """
    Dataset برای تسک‌های Image Restoration
    شامل تصاویر خراب (degraded) و تصاویر تمیز (clean)
    """
    def __init__(self, degraded_dir, clean_dir, patch_size=256, augment=True):
        """
        Args:
            degraded_dir: مسیر فولدر تصاویر خراب
            clean_dir: مسیر فولدر تصاویر تمیز
            patch_size: اندازه patch برای training (None برای استفاده از تصویر کامل)
            augment: آیا data augmentation انجام شود
        """
        self.degraded_dir = Path(degraded_dir)
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.augment = augment
        
        # لیست فایل‌های تصویر
        self.degraded_images = sorted([f for f in self.degraded_dir.iterdir() 
                                       if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        self.clean_images = sorted([f for f in self.clean_dir.iterdir() 
                                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        
        assert len(self.degraded_images) == len(self.clean_images), \
            "Number of degraded and clean images must match"
        
        print(f"Dataset loaded: {len(self.degraded_images)} image pairs")

    def __len__(self):
        return len(self.degraded_images)

    def __getitem__(self, idx):
        # بارگذاری تصاویر
        degraded_path = self.degraded_images[idx]
        clean_path = self.clean_images[idx]
        
        degraded = Image.open(degraded_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        
        # تبدیل به tensor
        degraded = TF.to_tensor(degraded)
        clean = TF.to_tensor(clean)
        
        # اطمینان از اندازه یکسان
        h, w = degraded.shape[1:]
        clean = TF.resize(clean, [h, w])
        
        # Data augmentation
        if self.augment:
            degraded, clean = self.augment_pair(degraded, clean)
        
        # برش patch اگر مشخص شده باشد
        if self.patch_size is not None:
            degraded, clean = self.random_crop(degraded, clean, self.patch_size)
        
        return degraded, clean

    def augment_pair(self, degraded, clean):
        """Data augmentation برای یک جفت تصویر"""
        # Random horizontal flip
        if random.random() > 0.5:
            degraded = TF.hflip(degraded)
            clean = TF.hflip(clean)
        
        # Random vertical flip
        if random.random() > 0.5:
            degraded = TF.vflip(degraded)
            clean = TF.vflip(clean)
        
        # Random rotation (90, 180, 270 degrees)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            degraded = TF.rotate(degraded, angle)
            clean = TF.rotate(clean, angle)
        
        return degraded, clean

    def random_crop(self, degraded, clean, crop_size):
        """برش تصادفی patch از تصویر"""
        _, h, w = degraded.shape
        
        # اگر تصویر کوچکتر از crop_size باشد
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            degraded = TF.pad(degraded, [0, 0, pad_w, pad_h], padding_mode='reflect')
            clean = TF.pad(clean, [0, 0, pad_w, pad_h], padding_mode='reflect')
            _, h, w = degraded.shape
        
        # برش تصادفی
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        degraded = TF.crop(degraded, top, left, crop_size, crop_size)
        clean = TF.crop(clean, top, left, crop_size, crop_size)
        
        return degraded, clean


class DeblurDataset(ImageRestorationDataset):
    """Dataset مخصوص Image Deblurring"""
    pass


class DenoiseDataset(ImageRestorationDataset):
    """Dataset مخصوص Image Denoising"""
    pass


class SyntheticDegradationDataset(Dataset):
    """
    Dataset برای ایجاد تخریب مصنوعی
    فقط تصاویر clean نیاز دارد و تخریب را خودش ایجاد می‌کند
    """
    def __init__(self, clean_dir, patch_size=256, augment=True, 
                 degradation_type='blur', degradation_params=None):
        """
        Args:
            clean_dir: مسیر فولدر تصاویر تمیز
            patch_size: اندازه patch
            augment: data augmentation
            degradation_type: نوع تخریب ('blur', 'noise', 'jpeg', 'combined')
            degradation_params: پارامترهای تخریب
        """
        self.clean_dir = Path(clean_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.degradation_type = degradation_type
        
        # پارامترهای پیش‌فرض
        self.degradation_params = degradation_params or {
            'blur_kernel_size': [7, 9, 11, 13, 15],
            'blur_sigma': [0.1, 3.0],
            'noise_sigma': [0, 50],
            'jpeg_quality': [30, 95]
        }
        
        self.clean_images = sorted([f for f in self.clean_dir.iterdir() 
                                    if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        
        print(f"Synthetic dataset loaded: {len(self.clean_images)} clean images")

    def __len__(self):
        return len(self.clean_images)

    def __getitem__(self, idx):
        # بارگذاری تصویر تمیز
        clean_path = self.clean_images[idx]
        clean = Image.open(clean_path).convert('RGB')
        clean = TF.to_tensor(clean)
        
        # ایجاد تخریب
        degraded = self.apply_degradation(clean.clone())
        
        # Data augmentation
        if self.augment:
            degraded, clean = self.augment_pair(degraded, clean)
        
        # برش patch
        if self.patch_size is not None:
            degraded, clean = self.random_crop(degraded, clean, self.patch_size)
        
        return degraded, clean

    def apply_degradation(self, image):
        """اعمال تخریب به تصویر"""
        if self.degradation_type == 'blur':
            return self.apply_blur(image)
        elif self.degradation_type == 'noise':
            return self.apply_noise(image)
        elif self.degradation_type == 'jpeg':
            return self.apply_jpeg(image)
        elif self.degradation_type == 'combined':
            # ترکیب تصادفی از تخریب‌ها
            degradations = [self.apply_blur, self.apply_noise, self.apply_jpeg]
            random.shuffle(degradations)
            for deg in degradations[:random.randint(1, 3)]:
                image = deg(image)
            return image
        else:
            return image

    def apply_blur(self, image):
        """اعمال blur به تصویر"""
        kernel_size = random.choice(self.degradation_params['blur_kernel_size'])
        sigma_min, sigma_max = self.degradation_params['blur_sigma']
        sigma = random.uniform(sigma_min, sigma_max)
        
        return TF.gaussian_blur(image, kernel_size, sigma)

    def apply_noise(self, image):
        """اعمال Gaussian noise"""
        sigma_min, sigma_max = self.degradation_params['noise_sigma']
        sigma = random.uniform(sigma_min, sigma_max) / 255.0
        
        noise = torch.randn_like(image) * sigma
        return torch.clamp(image + noise, 0, 1)

    def apply_jpeg(self, image):
        """شبیه‌سازی فشرده‌سازی JPEG"""
        quality_min, quality_max = self.degradation_params['jpeg_quality']
        quality = random.randint(quality_min, quality_max)
        
        # تبدیل به PIL، ذخیره با کیفیت پایین، و بازگشت به tensor
        import io
        img_pil = TF.to_pil_image(image)
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        img_pil = Image.open(buffer)
        return TF.to_tensor(img_pil)

    def augment_pair(self, degraded, clean):
        """Data augmentation"""
        if random.random() > 0.5:
            degraded = TF.hflip(degraded)
            clean = TF.hflip(clean)
        
        if random.random() > 0.5:
            degraded = TF.vflip(degraded)
            clean = TF.vflip(clean)
        
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            degraded = TF.rotate(degraded, angle)
            clean = TF.rotate(clean, angle)
        
        return degraded, clean

    def random_crop(self, degraded, clean, crop_size):
        """برش تصادفی"""
        _, h, w = degraded.shape
        
        if h < crop_size or w < crop_size:
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            degraded = TF.pad(degraded, [0, 0, pad_w, pad_h], padding_mode='reflect')
            clean = TF.pad(clean, [0, 0, pad_w, pad_h], padding_mode='reflect')
            _, h, w = degraded.shape
        
        top = random.randint(0, h - crop_size)
        left = random.randint(0, w - crop_size)
        
        degraded = TF.crop(degraded, top, left, crop_size, crop_size)
        clean = TF.crop(clean, top, left, crop_size, crop_size)
        
        return degraded, clean


def create_dataloaders(config):
    """ایجاد train و validation dataloaders"""
    
    if config['data'].get('synthetic', False):
        # استفاده از synthetic degradation
        train_dataset = SyntheticDegradationDataset(
            clean_dir=config['data']['train_clean'],
            patch_size=config['data']['patch_size'],
            augment=True,
            degradation_type=config['data'].get('degradation_type', 'combined')
        )
        
        val_dataset = SyntheticDegradationDataset(
            clean_dir=config['data']['val_clean'],
            patch_size=None,
            augment=False,
            degradation_type=config['data'].get('degradation_type', 'combined')
        )
    else:
        # استفاده از paired data
        train_dataset = ImageRestorationDataset(
            degraded_dir=config['data']['train_degraded'],
            clean_dir=config['data']['train_clean'],
            patch_size=config['data']['patch_size'],
            augment=True
        )
        
        val_dataset = ImageRestorationDataset(
            degraded_dir=config['data']['val_degraded'],
            clean_dir=config['data']['val_clean'],
            patch_size=None,
            augment=False
        )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # تست dataset
    print("Testing ImageRestorationDataset...")
    
    # فرض کنید داده‌ها در این مسیرها هستند
    dataset = ImageRestorationDataset(
        degraded_dir='./data/train/degraded',
        clean_dir='./data/train/clean',
        patch_size=256,
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # تست یک sample
    if len(dataset) > 0:
        degraded, clean = dataset[0]
        print(f"Degraded shape: {degraded.shape}")
        print(f"Clean shape: {clean.shape}")
        print(f"Degraded range: [{degraded.min():.3f}, {degraded.max():.3f}]")
        print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
