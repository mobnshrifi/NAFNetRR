"""
Utility Functions for Image Restoration
توابع کمکی برای پروژه
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from math import log10, sqrt
from skimage.metrics import structural_similarity as ssim_skimage
import os
from pathlib import Path


class AverageMeter:
    """محاسبه و ذخیره میانگین و مقدار فعلی"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_psnr(img1, img2, max_val=1.0):
    """
    محاسبه PSNR بین دو تصویر
    Args:
        img1, img2: تصاویر با shape (B, C, H, W) یا (C, H, W)
        max_val: مقدار ماکزیمم پیکسل (1.0 برای normalized images)
    Returns:
        PSNR value
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100.0
    
    psnr = 20 * log10(max_val) - 10 * log10(mse.item())
    return psnr


def calculate_ssim(img1, img2, max_val=1.0):
    """
    محاسبه SSIM بین دو تصویر
    Args:
        img1, img2: تصاویر با shape (B, C, H, W) یا (C, H, W)
        max_val: مقدار ماکزیمم پیکسل
    Returns:
        SSIM value
    """
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
    
    # تبدیل به numpy برای استفاده از skimage
    img1_np = img1[0].cpu().numpy().transpose(1, 2, 0)
    img2_np = img2[0].cpu().numpy().transpose(1, 2, 0)
    
    # محاسبه SSIM برای هر کانال
    if img1_np.shape[2] == 3:
        ssim_val = ssim_skimage(img1_np, img2_np, channel_axis=2, data_range=max_val)
    else:
        ssim_val = ssim_skimage(img1_np.squeeze(), img2_np.squeeze(), data_range=max_val)
    
    return ssim_val


def calculate_psnr_batch(img1, img2, max_val=1.0):
    """محاسبه PSNR برای یک batch"""
    batch_size = img1.size(0)
    psnr_values = []
    
    for i in range(batch_size):
        psnr = calculate_psnr(img1[i], img2[i], max_val)
        psnr_values.append(psnr)
    
    return np.mean(psnr_values)


def calculate_ssim_batch(img1, img2, max_val=1.0):
    """محاسبه SSIM برای یک batch"""
    batch_size = img1.size(0)
    ssim_values = []
    
    for i in range(batch_size):
        ssim_val = calculate_ssim(img1[i], img2[i], max_val)
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)


def save_checkpoint(state, filename='checkpoint.pth'):
    """ذخیره checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, checkpoint_path, device='cuda'):
    """بارگذاری checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def save_image(tensor, path):
    """
    ذخیره تصویر از tensor
    Args:
        tensor: تصویر با shape (C, H, W) یا (B, C, H, W)
        path: مسیر ذخیره
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    # Denormalize اگر لازم باشد
    img = tensor.cpu().clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    
    # تبدیل RGB به BGR برای OpenCV
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(path), img)


def create_exp_dir(exp_name, base_dir='./experiments'):
    """
    ایجاد دایرکتوری برای experiment
    Returns:
        exp_dir: مسیر experiment
    """
    base_dir = Path(base_dir)
    exp_dir = base_dir / exp_name
    
    # ایجاد زیرپوشه‌ها
    (exp_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'results').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'configs').mkdir(parents=True, exist_ok=True)
    
    return exp_dir


def count_parameters(model):
    """شمارش تعداد پارامترهای قابل آموزش"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_network(model, name='Model'):
    """چاپ اطلاعات شبکه"""
    num_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(f"Total parameters: {num_params:,}")
    print(f"{'='*50}\n")


class ProgressiveTraining:
    """
    کلاس برای آموزش پیشرونده
    ابتدا با iterations کم شروع و تدریجاً افزایش می‌دهیم
    """
    def __init__(self, max_iterations, warmup_epochs=10, total_epochs=100):
        self.max_iterations = max_iterations
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def get_num_iterations(self, epoch):
        """تعداد iterations برای epoch فعلی"""
        if epoch < self.warmup_epochs:
            # در warmup، از 1 تا max_iterations افزایش می‌دهیم
            return min(1 + epoch * (self.max_iterations - 1) // self.warmup_epochs, 
                      self.max_iterations)
        else:
            # بعد از warmup، از تعداد کامل استفاده می‌کنیم
            return self.max_iterations


def visualize_reasoning_process(model, image, save_dir, num_iterations=None):
    """
    Visualize کردن فرآیند reasoning
    تصاویر را در هر iteration ذخیره می‌کند
    """
    import matplotlib.pyplot as plt
    
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        # ذخیره تصویر ورودی
        save_image(image, save_dir / 'input.png')
        
        # اگر num_iterations مشخص نشده، از مقدار مدل استفاده کن
        if num_iterations is None:
            num_iterations = model.reasoning_iterations
        
        # تولید تصاویر با تعداد iterations مختلف
        results = []
        for i in range(1, num_iterations + 1):
            output = model(image.unsqueeze(0), reasoning_iters=i)
            results.append(output[0])
            save_image(output[0], save_dir / f'iter_{i}.png')
        
        # ایجاد یک visualization ترکیبی
        fig, axes = plt.subplots(1, num_iterations + 1, figsize=(4*(num_iterations+1), 4))
        
        # تصویر ورودی
        img_input = image.cpu().permute(1, 2, 0).numpy()
        axes[0].imshow(img_input)
        axes[0].set_title('Input')
        axes[0].axis('off')
        
        # تصاویر خروجی
        for i, result in enumerate(results):
            img_result = result.cpu().permute(1, 2, 0).numpy()
            axes[i+1].imshow(np.clip(img_result, 0, 1))
            axes[i+1].set_title(f'Iteration {i+1}')
            axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'reasoning_process.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()


def prepare_input(image_path, device='cuda'):
    """
    آماده‌سازی تصویر ورودی برای inference
    Args:
        image_path: مسیر تصویر
        device: دستگاه محاسباتی
    Returns:
        tensor: تصویر آماده شده
    """
    import torchvision.transforms as transforms
    from PIL import Image
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    tensor = transform(image).unsqueeze(0).to(device)
    
    return tensor


class EarlyStopping:
    """Early stopping برای جلوگیری از overfitting"""
    def __init__(self, patience=10, min_delta=0, mode='max'):
        """
        Args:
            patience: تعداد epoch های صبر
            min_delta: حداقل تغییر برای بهبود
            mode: 'max' برای metrics مثل PSNR، 'min' برای loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.is_improved(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

    def is_improved(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta


def log_results(results_dict, log_file):
    """ذخیره نتایج در فایل لاگ"""
    import json
    from datetime import datetime
    
    results_dict['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a') as f:
        json.dump(results_dict, f)
        f.write('\n')


if __name__ == '__main__':
    # تست توابع
    print("Testing utility functions...")
    
    # تست PSNR و SSIM
    img1 = torch.rand(1, 3, 256, 256)
    img2 = img1 + torch.randn_like(img1) * 0.01
    
    psnr = calculate_psnr(img1, img2)
    ssim_val = calculate_ssim(img1, img2)
    
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    
    # تست Progressive Training
    progressive = ProgressiveTraining(max_iterations=5, warmup_epochs=10)
    for epoch in range(15):
        iters = progressive.get_num_iterations(epoch)
        print(f"Epoch {epoch}: {iters} iterations")
    
    print("\nUtility functions work correctly!")
