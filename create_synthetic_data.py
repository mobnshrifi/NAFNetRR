"""
Create Synthetic Degraded Images
ایجاد تصاویر خراب مصنوعی برای آموزش
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import cv2
from tqdm import tqdm
import random


class SyntheticDegradation:
    """کلاس برای ایجاد تخریب‌های مصنوعی"""
    
    def __init__(self, degradation_type='combined'):
        self.degradation_type = degradation_type
    
    def apply_blur(self, image, kernel_size=None, sigma=None):
        """اعمال Motion Blur یا Gaussian Blur"""
        if kernel_size is None:
            kernel_size = random.choice([7, 9, 11, 13, 15])
        if sigma is None:
            sigma = random.uniform(0.5, 3.0)
        
        # Gaussian Blur
        if random.random() > 0.5:
            return image.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        # Motion Blur
        else:
            # تبدیل به numpy
            img_np = np.array(image)
            
            # ایجاد Motion Blur kernel
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            
            # چرخش تصادفی kernel
            angle = random.uniform(0, 360)
            M = cv2.getRotationMatrix2D((kernel_size/2, kernel_size/2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            
            # اعمال blur
            blurred = cv2.filter2D(img_np, -1, kernel)
            
            return Image.fromarray(blurred)
    
    def apply_noise(self, image, sigma=None):
        """اعمال Gaussian Noise"""
        if sigma is None:
            sigma = random.uniform(5, 50)
        
        img_np = np.array(image).astype(np.float32)
        
        # افزودن Gaussian noise
        noise = np.random.normal(0, sigma, img_np.shape)
        noisy = img_np + noise
        
        # Clip به بازه [0, 255]
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy)
    
    def apply_jpeg_compression(self, image, quality=None):
        """شبیه‌سازی فشرده‌سازی JPEG"""
        if quality is None:
            quality = random.randint(30, 90)
        
        # ذخیره و بارگذاری با کیفیت پایین
        import io
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).copy()
        
        return compressed
    
    def apply_downsampling(self, image, scale=None):
        """کاهش رزولوشن و افزایش مجدد"""
        if scale is None:
            scale = random.uniform(0.3, 0.7)
        
        w, h = image.size
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Downscale
        downsampled = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Upscale به اندازه اصلی
        upsampled = downsampled.resize((w, h), Image.BILINEAR)
        
        return upsampled
    
    def apply_combined(self, image):
        """ترکیب تصادفی از تخریب‌ها"""
        degradations = []
        
        # انتخاب تصادفی تخریب‌ها
        if random.random() > 0.3:
            degradations.append(self.apply_blur)
        if random.random() > 0.5:
            degradations.append(self.apply_noise)
        if random.random() > 0.6:
            degradations.append(self.apply_jpeg_compression)
        if random.random() > 0.7:
            degradations.append(self.apply_downsampling)
        
        # حداقل یک تخریب
        if len(degradations) == 0:
            degradations.append(random.choice([
                self.apply_blur,
                self.apply_noise,
                self.apply_jpeg_compression
            ]))
        
        # اعمال تخریب‌ها
        result = image
        random.shuffle(degradations)
        for deg_func in degradations:
            result = deg_func(result)
        
        return result
    
    def degrade(self, image):
        """اعمال تخریب بر اساس نوع انتخابی"""
        if self.degradation_type == 'blur':
            return self.apply_blur(image)
        elif self.degradation_type == 'noise':
            return self.apply_noise(image)
        elif self.degradation_type == 'jpeg':
            return self.apply_jpeg_compression(image)
        elif self.degradation_type == 'downsample':
            return self.apply_downsampling(image)
        elif self.degradation_type == 'combined':
            return self.apply_combined(image)
        else:
            raise ValueError(f"Unknown degradation type: {self.degradation_type}")


def process_folder(input_folder, output_folder, degradation_type='combined', num_augmentations=1):
    """
    پردازش یک فولدر از تصاویر و ایجاد نسخه‌های خراب
    
    Args:
        input_folder: فولدر تصاویر تمیز
        output_folder: فولدر خروجی
        degradation_type: نوع تخریب
        num_augmentations: تعداد نسخه‌های خراب برای هر تصویر
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # لیست تصاویر
    image_files = sorted([f for f in input_folder.iterdir() 
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    
    print(f"Found {len(image_files)} images")
    print(f"Degradation type: {degradation_type}")
    print(f"Augmentations per image: {num_augmentations}")
    
    degrader = SyntheticDegradation(degradation_type)
    
    total_images = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # بارگذاری تصویر
            image = Image.open(img_path).convert('RGB')
            
            # ایجاد چندین نسخه خراب
            for aug_idx in range(num_augmentations):
                # اعمال تخریب
                degraded = degrader.degrade(image)
                
                # تعیین نام فایل خروجی
                if num_augmentations > 1:
                    output_name = f"{img_path.stem}_aug{aug_idx}{img_path.suffix}"
                else:
                    output_name = img_path.name
                
                output_path = output_folder / output_name
                
                # ذخیره
                degraded.save(output_path)
                total_images += 1
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print(f"\n✓ Processing completed!")
    print(f"Total degraded images created: {total_images}")


def create_paired_dataset(clean_folder, degraded_folder, val_split=0.1, degradation_type='combined'):
    """
    ایجاد یک دیتاست paired کامل با train/val split
    
    Args:
        clean_folder: فولدر تصاویر تمیز
        degraded_folder: فولدر برای ذخیره تصاویر خراب
        val_split: نسبت داده‌های validation
        degradation_type: نوع تخریب
    """
    clean_folder = Path(clean_folder)
    
    # ایجاد ساختار فولدرها
    train_degraded = Path(degraded_folder) / 'train' / 'degraded'
    train_clean = Path(degraded_folder) / 'train' / 'clean'
    val_degraded = Path(degraded_folder) / 'val' / 'degraded'
    val_clean = Path(degraded_folder) / 'val' / 'clean'
    
    for folder in [train_degraded, train_clean, val_degraded, val_clean]:
        folder.mkdir(parents=True, exist_ok=True)
    
    # لیست تصاویر
    image_files = sorted([f for f in clean_folder.iterdir() 
                         if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
    
    # تقسیم train/val
    num_val = int(len(image_files) * val_split)
    random.shuffle(image_files)
    val_files = image_files[:num_val]
    train_files = image_files[num_val:]
    
    print(f"Total images: {len(image_files)}")
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    degrader = SyntheticDegradation(degradation_type)
    
    # پردازش train
    print("\nProcessing training set...")
    for img_path in tqdm(train_files):
        try:
            image = Image.open(img_path).convert('RGB')
            
            # کپی تصویر تمیز
            image.save(train_clean / img_path.name)
            
            # ایجاد تصویر خراب
            degraded = degrader.degrade(image)
            degraded.save(train_degraded / img_path.name)
        
        except Exception as e:
            print(f"Error: {e}")
    
    # پردازش val
    print("\nProcessing validation set...")
    for img_path in tqdm(val_files):
        try:
            image = Image.open(img_path).convert('RGB')
            
            # کپی تصویر تمیز
            image.save(val_clean / img_path.name)
            
            # ایجاد تصویر خراب
            degraded = degrader.degrade(image)
            degraded.save(val_degraded / img_path.name)
        
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✓ Dataset created successfully!")
    print(f"Location: {degraded_folder}")


def main():
    parser = argparse.ArgumentParser(description='Create Synthetic Degraded Images')
    parser.add_argument('--input', type=str, required=True,
                       help='Input folder with clean images')
    parser.add_argument('--output', type=str, required=True,
                       help='Output folder for degraded images')
    parser.add_argument('--type', type=str, default='combined',
                       choices=['blur', 'noise', 'jpeg', 'downsample', 'combined'],
                       help='Degradation type')
    parser.add_argument('--augmentations', type=int, default=1,
                       help='Number of degraded versions per image')
    parser.add_argument('--create-paired', action='store_true',
                       help='Create a complete paired dataset with train/val split')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (for --create-paired)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Synthetic Degradation Generator")
    print("="*60 + "\n")
    
    if args.create_paired:
        create_paired_dataset(
            args.input,
            args.output,
            args.val_split,
            args.type
        )
    else:
        process_folder(
            args.input,
            args.output,
            args.type,
            args.augmentations
        )


if __name__ == '__main__':
    main()
