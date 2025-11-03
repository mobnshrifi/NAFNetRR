"""
Setup Script for NAFNet-RR Project
اسکریپت راه‌اندازی اولیه پروژه
"""

import os
from pathlib import Path
import argparse
import json


def create_directory_structure():
    """ایجاد ساختار دایرکتوری پروژه"""
    
    directories = [
        'data/train/degraded',
        'data/train/clean',
        'data/val/degraded',
        'data/val/clean',
        'data/test/degraded',
        'data/test/clean',
        'experiments/checkpoints',
        'experiments/logs',
        'experiments/results',
        'experiments/configs',
        'pretrained_models',
        'test_images',
        'demo_results',
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")
    
    print("\n✓ Directory structure created successfully!")


def create_sample_config(config_path='config.yaml'):
    """ایجاد فایل config نمونه"""
    
    sample_config = """# Configuration for NAFNet-RR Training

# نام experiment
exp_name: 'NAFNet_RR_experiment'

# Random seed
seed: 42

# Model configuration
model:
  img_channel: 3
  width: 64
  middle_blk_num: 12
  enc_blk_nums: [2, 2, 4, 8]
  dec_blk_nums: [2, 2, 2, 2]
  reasoning_iterations: 3
  reasoning_positions:
    - 'middle'
    - 'decoder'

# Data configuration
data:
  train_degraded: './data/train/degraded'
  train_clean: './data/train/clean'
  val_degraded: './data/val/degraded'
  val_clean: './data/val/clean'
  
  synthetic: false
  degradation_type: 'combined'
  patch_size: 256
  augment: true

# Training configuration
training:
  batch_size: 8
  epochs: 500
  lr: 0.0002
  lr_min: 0.000001
  weight_decay: 0.0001
  
  scheduler: 'cosine'
  grad_clip: 1.0
  random_iterations: false
  
  log_interval: 50
  val_interval: 1
  save_interval: 10
  save_img_interval: 5
  
  num_workers: 8

# Paths
checkpoint_dir: './experiments/checkpoints'
log_dir: './experiments/logs'
result_dir: './experiments/results'

# Resume training
resume: null
"""
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(sample_config)
    
    print(f"\n✓ Sample config created: {config_path}")


def create_gitignore():
    """ایجاد فایل .gitignore"""
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/
*.jpg
*.jpeg
*.png
*.bmp
*.gif

# Experiments
experiments/
!experiments/.gitkeep

# Checkpoints
*.pth
*.pt
pretrained_models/

# Logs
*.log
logs/
tensorboard/

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✓ .gitignore created")


def create_quick_start_script():
    """ایجاد اسکریپت quick start"""
    
    script_content = """#!/bin/bash
# Quick Start Script for NAFNet-RR

echo "NAFNet-RR Quick Start"
echo "====================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate  # For Linux/Mac
# venv\\Scripts\\activate  # For Windows

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directory structure
echo "Setting up project structure..."
python setup_project.py

echo ""
echo "✓ Setup completed!"
echo ""
echo "Next steps:"
echo "1. Place your training data in data/train/"
echo "2. Edit config.yaml according to your needs"
echo "3. Start training: python train.py --config config.yaml"
echo ""
"""
    
    with open('quick_start.sh', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('quick_start.sh', 0o755)
    
    print("✓ Quick start script created: quick_start.sh")


def create_project_info():
    """ایجاد فایل اطلاعات پروژه"""
    
    info = {
        "project": "NAFNet-RR",
        "description": "NAFNet with Recurrent Reasoning for Image Restoration",
        "version": "1.0.0",
        "author": "Your Name",
        "files": {
            "model": "NAFNet_RR_model.py",
            "train": "train.py",
            "test": "test.py",
            "demo": "demo.py",
            "dataset": "dataset.py",
            "utils": "utils.py",
            "config": "config.yaml"
        },
        "structure": {
            "data": "Training and validation data",
            "experiments": "Checkpoints, logs, and results",
            "pretrained_models": "Pre-trained model weights",
            "test_images": "Sample images for testing",
            "demo_results": "Demo outputs"
        },
        "quick_commands": {
            "train": "python train.py --config config.yaml",
            "test_single": "python test.py --config config.yaml --checkpoint model.pth --mode single --input image.png",
            "test_folder": "python test.py --config config.yaml --checkpoint model.pth --mode folder --input ./test_images/ --output ./results/",
            "demo": "python demo.py --checkpoint model.pth --mode quick --input image.png"
        }
    }
    
    with open('project_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    print("✓ Project info created: project_info.json")


def check_dependencies():
    """بررسی نصب وابستگی‌ها"""
    
    required_packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'Pillow',
        'numpy',
        'tensorboard',
        'pyyaml',
        'tqdm'
    ]
    
    print("\nChecking dependencies...")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
    else:
        print("\n✓ All dependencies are installed!")


def print_usage_guide():
    """چاپ راهنمای استفاده"""
    
    guide = """
╔══════════════════════════════════════════════════════════════════╗
║                    NAFNet-RR Usage Guide                         ║
╚══════════════════════════════════════════════════════════════════╝

📁 PROJECT STRUCTURE
├── data/                  # Dataset directory
│   ├── train/            # Training data
│   └── val/              # Validation data
├── experiments/          # Training outputs
├── pretrained_models/    # Pre-trained weights
└── test_images/         # Test samples

🚀 QUICK START

1. Prepare your data:
   Place degraded images in: data/train/degraded/
   Place clean images in:    data/train/clean/

2. Configure the model:
   Edit config.yaml to set model parameters and data paths

3. Start training:
   python train.py --config config.yaml

4. Monitor training:
   tensorboard --logdir experiments/logs

5. Test the model:
   python test.py --config config.yaml --checkpoint experiments/checkpoints/best_model.pth --mode single --input test.png

📊 COMMON COMMANDS

Training:
  python train.py --config config.yaml
  python train.py --config config.yaml --resume experiments/checkpoints/latest.pth

Testing:
  # Single image
  python test.py --config config.yaml --checkpoint model.pth --mode single --input image.png
  
  # Folder
  python test.py --config config.yaml --checkpoint model.pth --mode folder --input ./test_images/ --output ./results/
  
  # Evaluation
  python test.py --config config.yaml --checkpoint model.pth --mode eval --input ./data/test/degraded --clean ./data/test/clean

Demo:
  python demo.py --checkpoint model.pth --mode quick --input image.png
  python demo.py --checkpoint model.pth --mode interactive

💡 TIPS

• Start with a small model (width=32) for testing
• Use synthetic degradation if you don't have paired data
• Monitor validation metrics to prevent overfitting
• Adjust batch_size based on your GPU memory
• Use more iterations for better quality (slower inference)

📚 For more information, check README.md

"""
    
    print(guide)


def main():
    parser = argparse.ArgumentParser(description='Setup NAFNet-RR Project')
    parser.add_argument('--full', action='store_true', 
                       help='Full setup including all files')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies only')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("NAFNet-RR Project Setup")
    print("="*70 + "\n")
    
    if args.check_deps:
        check_dependencies()
        return
    
    # ایجاد ساختار دایرکتوری
    create_directory_structure()
    
    if args.full:
        # ایجاد فایل‌های کمکی
        create_sample_config()
        create_gitignore()
        create_quick_start_script()
        create_project_info()
    
    # بررسی وابستگی‌ها
    check_dependencies()
    
    # نمایش راهنما
    print_usage_guide()
    
    print("\n✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Place your data in the data/ directory")
    print("2. Edit config.yaml")
    print("3. Run: python train.py --config config.yaml")
    print()


if __name__ == '__main__':
    main()
