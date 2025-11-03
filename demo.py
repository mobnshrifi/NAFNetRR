"""
Demo Script for NAFNet-RR
اسکریپت نمایشی ساده برای تست سریع مدل
"""

import torch
import argparse
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from NAFNet_RR_model import NAFNetRR, count_parameters
from utils import calculate_psnr, calculate_ssim, save_image


def load_model(checkpoint_path, device='cuda'):
    """بارگذاری مدل از checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    # ساخت مدل
    model = NAFNetRR(
        img_channel=config.get('model', {}).get('img_channel', 3),
        width=config.get('model', {}).get('width', 64),
        middle_blk_num=config.get('model', {}).get('middle_blk_num', 12),
        enc_blk_nums=config.get('model', {}).get('enc_blk_nums', [2, 2, 4, 8]),
        dec_blk_nums=config.get('model', {}).get('dec_blk_nums', [2, 2, 2, 2]),
        reasoning_iterations=config.get('model', {}).get('reasoning_iterations', 3),
        reasoning_positions=config.get('model', {}).get('reasoning_positions', ['middle', 'decoder'])
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Reasoning iterations: {model.reasoning_iterations}")
    
    return model


def quick_demo(image_path, checkpoint_path, output_dir='./demo_results'):
    """
    نمایش سریع قابلیت‌های مدل
    شامل: پردازش با iterations مختلف و نمایش نتایج
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ایجاد فولدر خروجی
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # بارگذاری مدل
    model = load_model(checkpoint_path, device)
    
    # بارگذاری تصویر
    print(f"\nProcessing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    input_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
    
    print(f"Image size: {image.size}")
    
    # پردازش با iterations مختلف
    max_iters = model.reasoning_iterations
    results = []
    times = []
    
    print(f"\nProcessing with different numbers of iterations...")
    with torch.no_grad():
        for num_iters in range(1, max_iters + 1):
            start_time = time.time()
            output = model(input_tensor, reasoning_iters=num_iters)
            inference_time = time.time() - start_time
            
            results.append(output[0].cpu())
            times.append(inference_time)
            
            print(f"  Iterations {num_iters}: {inference_time*1000:.1f} ms")
            
            # ذخیره تصویر
            save_image(output[0], output_dir / f'result_iter_{num_iters}.png')
    
    # ذخیره تصویر ورودی
    save_image(input_tensor[0], output_dir / 'input.png')
    
    # ایجاد visualization
    create_comparison_plot(
        input_tensor[0].cpu(),
        results,
        times,
        output_dir / 'comparison.png'
    )
    
    print(f"\nResults saved to {output_dir}")
    
    return results, times


def create_comparison_plot(input_img, results, times, save_path):
    """ایجاد نمودار مقایسه"""
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(4*(num_results+1), 4))
    
    # تصویر ورودی
    img_input = input_img.permute(1, 2, 0).numpy()
    axes[0].imshow(np.clip(img_input, 0, 1))
    axes[0].set_title('Input\n(Degraded)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # نتایج با iterations مختلف
    for i, (result, time_ms) in enumerate(zip(results, times)):
        img_result = result.permute(1, 2, 0).numpy()
        axes[i+1].imshow(np.clip(img_result, 0, 1))
        axes[i+1].set_title(f'Iteration {i+1}\n{time_ms*1000:.1f} ms', 
                           fontsize=12, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {save_path}")


def interactive_demo(checkpoint_path):
    """
    نسخه تعاملی demo که کاربر می‌تواند تصویر و تعداد iterations را انتخاب کند
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # بارگذاری مدل
    model = load_model(checkpoint_path, device)
    
    print("\n" + "="*60)
    print("Interactive Demo - NAFNet-RR")
    print("="*60)
    
    while True:
        # دریافت مسیر تصویر
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() == 'quit':
            print("Exiting demo...")
            break
        
        if not Path(image_path).exists():
            print(f"Error: File not found: {image_path}")
            continue
        
        try:
            # بارگذاری تصویر
            image = Image.open(image_path).convert('RGB')
            input_tensor = TF.to_tensor(image).unsqueeze(0).to(device)
            
            # دریافت تعداد iterations
            max_iters = model.reasoning_iterations
            iters_input = input(f"Number of iterations (1-{max_iters}, default={max_iters}): ").strip()
            
            if iters_input:
                num_iters = int(iters_input)
                num_iters = max(1, min(num_iters, max_iters))
            else:
                num_iters = max_iters
            
            # پردازش
            print(f"\nProcessing with {num_iters} iterations...")
            start_time = time.time()
            
            with torch.no_grad():
                output = model(input_tensor, reasoning_iters=num_iters)
            
            inference_time = time.time() - start_time
            
            # ذخیره نتیجه
            output_path = Path(image_path).parent / f"{Path(image_path).stem}_restored.png"
            save_image(output[0], output_path)
            
            print(f"✓ Processing completed in {inference_time*1000:.1f} ms")
            print(f"✓ Result saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing image: {e}")


def speed_test(checkpoint_path, image_size=(256, 256), num_runs=50):
    """
    تست سرعت مدل
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # بارگذاری مدل
    model = load_model(checkpoint_path, device)
    
    print("\n" + "="*60)
    print("Speed Test")
    print("="*60)
    print(f"Image size: {image_size}")
    print(f"Number of runs: {num_runs}")
    
    # تصویر تستی
    dummy_input = torch.randn(1, 3, *image_size).to(device)
    
    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # بنچمارک
    max_iters = model.reasoning_iterations
    
    print("\nRunning speed test...")
    results = {}
    
    for num_iters in range(1, max_iters + 1):
        times = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input, reasoning_iters=num_iters)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        results[num_iters] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': fps
        }
        
        print(f"\nIterations {num_iters}:")
        print(f"  Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Demo for NAFNet-RR')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'interactive', 'speed'],
                       help='Demo mode')
    parser.add_argument('--input', type=str, help='Input image path (for quick mode)')
    parser.add_argument('--output', type=str, default='./demo_results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        if not args.input:
            print("Error: --input is required for quick mode")
            return
        quick_demo(args.input, args.checkpoint, args.output)
    
    elif args.mode == 'interactive':
        interactive_demo(args.checkpoint)
    
    elif args.mode == 'speed':
        speed_test(args.checkpoint)


if __name__ == '__main__':
    # اگر مستقیم اجرا شد
    import sys
    if len(sys.argv) == 1:
        print("Demo Script for NAFNet-RR")
        print("\nUsage examples:")
        print("  Quick demo:")
        print("    python demo.py --checkpoint model.pth --mode quick --input image.png")
        print("\n  Interactive demo:")
        print("    python demo.py --checkpoint model.pth --mode interactive")
        print("\n  Speed test:")
        print("    python demo.py --checkpoint model.pth --mode speed")
    else:
        main()
