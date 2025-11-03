"""
Test Script for NAFNet-RR
اسکریپت تست و inference برای مدل NAFNet-RR
"""

import os
import torch
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from NAFNet_RR_model import NAFNetRR
from utils import calculate_psnr, calculate_ssim, save_image, AverageMeter, visualize_reasoning_process


class Tester:
    def __init__(self, config, checkpoint_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        self.model = NAFNetRR(
            img_channel=config['model']['img_channel'],
            width=config['model']['width'],
            middle_blk_num=config['model']['middle_blk_num'],
            enc_blk_nums=config['model']['enc_blk_nums'],
            dec_blk_nums=config['model']['dec_blk_nums'],
            reasoning_iterations=config['model']['reasoning_iterations'],
            reasoning_positions=config['model']['reasoning_positions']
        ).to(self.device)
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully! Best PSNR: {checkpoint.get('best_psnr', 'N/A')}")
        print(f"Device: {self.device}")

    @torch.no_grad()
    def test_single_image(self, image_path, output_path=None, num_iterations=None):
        """
        تست یک تصویر منفرد
        Args:
            image_path: مسیر تصویر ورودی
            output_path: مسیر ذخیره (اگر None باشد، کنار تصویر ورودی ذخیره می‌شود)
            num_iterations: تعداد iterations (None = استفاده از default)
        """
        # بارگذاری تصویر
        image = Image.open(image_path).convert('RGB')
        input_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
        
        # Inference
        start_time = time.time()
        output_tensor = self.model(input_tensor, reasoning_iters=num_iterations)
        inference_time = time.time() - start_time
        
        # ذخیره تصویر
        if output_path is None:
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_restored.png"
        
        save_image(output_tensor[0], output_path)
        
        print(f"Processed: {image_path}")
        print(f"Output saved to: {output_path}")
        print(f"Inference time: {inference_time:.3f}s")
        
        return output_tensor

    @torch.no_grad()
    def test_folder(self, input_folder, output_folder, num_iterations=None):
        """
        تست یک فولدر از تصاویر
        Args:
            input_folder: مسیر فولدر ورودی
            output_folder: مسیر فولدر خروجی
            num_iterations: تعداد iterations
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # لیست تصاویر
        image_files = sorted([f for f in input_folder.iterdir() 
                            if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        
        print(f"Found {len(image_files)} images")
        
        total_time = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # بارگذاری و پردازش
            image = Image.open(img_path).convert('RGB')
            input_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            
            start_time = time.time()
            output_tensor = self.model(input_tensor, reasoning_iters=num_iterations)
            inference_time = time.time() - start_time
            total_time += inference_time
            
            # ذخیره
            output_path = output_folder / img_path.name
            save_image(output_tensor[0], output_path)
        
        avg_time = total_time / len(image_files)
        print(f"\nProcessing completed!")
        print(f"Average inference time: {avg_time:.3f}s per image")
        print(f"Total time: {total_time:.2f}s")

    @torch.no_grad()
    def test_with_ground_truth(self, degraded_folder, clean_folder, output_folder=None, num_iterations=None):
        """
        تست با ground truth برای محاسبه metrics
        Args:
            degraded_folder: مسیر تصاویر خراب
            clean_folder: مسیر تصاویر تمیز
            output_folder: مسیر ذخیره نتایج (اختیاری)
            num_iterations: تعداد iterations
        """
        degraded_folder = Path(degraded_folder)
        clean_folder = Path(clean_folder)
        
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # لیست تصاویر
        degraded_images = sorted([f for f in degraded_folder.iterdir() 
                                 if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        clean_images = sorted([f for f in clean_folder.iterdir() 
                              if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']])
        
        assert len(degraded_images) == len(clean_images), "Number of images must match"
        
        print(f"Testing on {len(degraded_images)} image pairs")
        
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()
        time_meter = AverageMeter()
        
        results = []
        
        for deg_path, clean_path in tqdm(zip(degraded_images, clean_images), 
                                        total=len(degraded_images),
                                        desc="Evaluating"):
            # بارگذاری تصاویر
            degraded = Image.open(deg_path).convert('RGB')
            clean = Image.open(clean_path).convert('RGB')
            
            degraded_tensor = TF.to_tensor(degraded).unsqueeze(0).to(self.device)
            clean_tensor = TF.to_tensor(clean).unsqueeze(0).to(self.device)
            
            # Inference
            start_time = time.time()
            restored_tensor = self.model(degraded_tensor, reasoning_iters=num_iterations)
            inference_time = time.time() - start_time
            
            # محاسبه metrics
            psnr = calculate_psnr(restored_tensor, clean_tensor)
            ssim = calculate_ssim(restored_tensor, clean_tensor)
            
            psnr_meter.update(psnr)
            ssim_meter.update(ssim)
            time_meter.update(inference_time)
            
            results.append({
                'image': deg_path.name,
                'psnr': psnr,
                'ssim': ssim,
                'time': inference_time
            })
            
            # ذخیره تصویر بازسازی شده
            if output_folder:
                save_image(restored_tensor[0], output_folder / deg_path.name)
        
        # چاپ نتایج
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Average PSNR: {psnr_meter.avg:.4f} dB")
        print(f"Average SSIM: {ssim_meter.avg:.4f}")
        print(f"Average Time: {time_meter.avg:.4f}s")
        print("="*60)
        
        # ذخیره نتایج جزئی
        if output_folder:
            import json
            results_file = output_folder / 'results.json'
            with open(results_file, 'w') as f:
                json.dump({
                    'average': {
                        'psnr': psnr_meter.avg,
                        'ssim': ssim_meter.avg,
                        'time': time_meter.avg
                    },
                    'per_image': results
                }, f, indent=4)
            print(f"\nDetailed results saved to {results_file}")
        
        return psnr_meter.avg, ssim_meter.avg

    @torch.no_grad()
    def compare_iterations(self, image_path, output_folder, max_iterations=None):
        """
        مقایسه نتایج با تعداد iterations مختلف
        Args:
            image_path: مسیر تصویر ورودی
            output_folder: مسیر ذخیره نتایج
            max_iterations: حداکثر تعداد iterations
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # بارگذاری تصویر
        image = Image.open(image_path).convert('RGB')
        input_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
        
        if max_iterations is None:
            max_iterations = self.model.reasoning_iterations
        
        print(f"Comparing results with different number of iterations (1 to {max_iterations})")
        
        results = {}
        
        for num_iters in range(1, max_iterations + 1):
            start_time = time.time()
            output_tensor = self.model(input_tensor, reasoning_iters=num_iters)
            inference_time = time.time() - start_time
            
            # ذخیره تصویر
            output_path = output_folder / f"iter_{num_iters}.png"
            save_image(output_tensor[0], output_path)
            
            results[num_iters] = {
                'time': inference_time,
                'path': str(output_path)
            }
            
            print(f"Iteration {num_iters}: {inference_time:.3f}s - saved to {output_path}")
        
        # Visualize کردن فرآیند reasoning
        visualize_reasoning_process(
            self.model, 
            input_tensor[0], 
            output_folder / 'visualization',
            num_iterations=max_iterations
        )
        
        return results

    @torch.no_grad()
    def benchmark_speed(self, image_size=(256, 256), num_runs=100, warmup=10):
        """
        بنچمارک سرعت مدل
        Args:
            image_size: اندازه تصویر تست
            num_runs: تعداد دفعات اجرا
            warmup: تعداد دفعات warmup
        """
        print(f"Benchmarking model speed on {image_size} images...")
        
        # تصویر تستی
        dummy_input = torch.randn(1, 3, *image_size).to(self.device)
        
        # Warmup
        print(f"Warming up ({warmup} runs)...")
        for _ in range(warmup):
            _ = self.model(dummy_input)
        
        # بنچمارک برای تعداد iterations مختلف
        max_iters = self.model.reasoning_iterations
        
        print(f"\nRunning benchmark ({num_runs} runs per configuration)...")
        
        for num_iters in range(1, max_iters + 1):
            times = []
            
            for _ in tqdm(range(num_runs), desc=f"Iterations {num_iters}"):
                start_time = time.time()
                _ = self.model(dummy_input, reasoning_iters=num_iters)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            fps = 1.0 / avg_time
            
            print(f"\nIterations {num_iters}:")
            print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
            print(f"  FPS: {fps:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Test NAFNet-RR for Image Restoration')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'folder', 'eval', 'compare', 'benchmark'],
                       help='Test mode')
    parser.add_argument('--input', type=str, help='Input image/folder path')
    parser.add_argument('--output', type=str, help='Output image/folder path')
    parser.add_argument('--clean', type=str, help='Clean images folder (for eval mode)')
    parser.add_argument('--iterations', type=int, default=None, 
                       help='Number of reasoning iterations (default: use model default)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create tester
    tester = Tester(config, args.checkpoint)
    
    # Run test based on mode
    if args.mode == 'single':
        if not args.input:
            raise ValueError("--input is required for single image mode")
        tester.test_single_image(args.input, args.output, args.iterations)
    
    elif args.mode == 'folder':
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for folder mode")
        tester.test_folder(args.input, args.output, args.iterations)
    
    elif args.mode == 'eval':
        if not args.input or not args.clean:
            raise ValueError("--input and --clean are required for eval mode")
        tester.test_with_ground_truth(args.input, args.clean, args.output, args.iterations)
    
    elif args.mode == 'compare':
        if not args.input or not args.output:
            raise ValueError("--input and --output are required for compare mode")
        tester.compare_iterations(args.input, args.output, args.iterations)
    
    elif args.mode == 'benchmark':
        tester.benchmark_speed()


if __name__ == '__main__':
    main()
    
