"""
Training Script for NAFNet-RR
اسکریپت آموزش مدل NAFNet با Recurrent Reasoning
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import yaml
import time
from pathlib import Path

from NAFNet_RR_model import NAFNetRR
from dataset import ImageRestorationDataset
from utils import AverageMeter, calculate_psnr, calculate_ssim, save_checkpoint, load_checkpoint


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)


class ProgressiveLoss(nn.Module):
    """
    Loss با وزن‌های متفاوت برای iterations مختلف
    Iterations اولیه کمتر و iterations آخر بیشتر وزن دارند
    """
    def __init__(self, num_iterations=3, base_loss=None):
        super().__init__()
        self.num_iterations = num_iterations
        self.base_loss = base_loss if base_loss else CharbonnierLoss()
        
        # وزن‌های افزایشی برای هر iteration
        weights = torch.linspace(0.5, 1.0, num_iterations)
        self.register_buffer('weights', weights / weights.sum())

    def forward(self, pred, target):
        return self.base_loss(pred, target)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
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
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
        # Loss function
        self.criterion = ProgressiveLoss(
            num_iterations=config['model']['reasoning_iterations'],
            base_loss=CharbonnierLoss()
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Learning rate scheduler
        if config['training']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['training']['lr_min']
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['lr_decay_steps'],
                gamma=config['training']['lr_decay_gamma']
            )
        
        # Data loaders
        train_dataset = ImageRestorationDataset(
            degraded_dir=config['data']['train_degraded'],
            clean_dir=config['data']['train_clean'],
            patch_size=config['data']['patch_size'],
            augment=True
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
        
        val_dataset = ImageRestorationDataset(
            degraded_dir=config['data']['val_degraded'],
            clean_dir=config['data']['val_clean'],
            patch_size=None,  # Full image for validation
            augment=False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Training state
        self.start_epoch = 0
        self.best_psnr = 0
        self.global_step = 0
        
        # Resume from checkpoint if specified
        if config.get('resume'):
            self.load_checkpoint(config['resume'])

    def train_epoch(self, epoch):
        self.model.train()
        
        losses = AverageMeter()
        psnrs = AverageMeter()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}")
        
        for degraded, clean in pbar:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            # در حین آموزش، از تعداد متفاوتی از iterations استفاده می‌کنیم
            # برای robust شدن مدل
            if self.config['training'].get('random_iterations', False):
                num_iters = torch.randint(1, self.config['model']['reasoning_iterations'] + 1, (1,)).item()
            else:
                num_iters = None  # استفاده از تعداد پیش‌فرض
            
            restored = self.model(degraded, reasoning_iters=num_iters)
            
            # Calculate loss
            loss = self.criterion(restored, clean)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                psnr = calculate_psnr(restored, clean)
            
            # Update meters
            losses.update(loss.item())
            psnrs.update(psnr)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'psnr': f'{psnrs.avg:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to tensorboard
            if self.global_step % self.config['training']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/psnr', psnr, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        return losses.avg, psnrs.avg

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        
        losses = AverageMeter()
        psnrs = AverageMeter()
        ssims = AverageMeter()
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for degraded, clean in pbar:
            degraded = degraded.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass با تعداد کامل iterations
            restored = self.model(degraded)
            
            # Calculate loss and metrics
            loss = self.criterion(restored, clean)
            psnr = calculate_psnr(restored, clean)
            ssim = calculate_ssim(restored, clean)
            
            losses.update(loss.item())
            psnrs.update(psnr)
            ssims.update(ssim)
            
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'psnr': f'{psnrs.avg:.2f}',
                'ssim': f'{ssims.avg:.4f}'
            })
        
        # Log to tensorboard
        self.writer.add_scalar('val/loss', losses.avg, epoch)
        self.writer.add_scalar('val/psnr', psnrs.avg, epoch)
        self.writer.add_scalar('val/ssim', ssims.avg, epoch)
        
        # Save sample images
        if epoch % self.config['training']['save_img_interval'] == 0:
            self.writer.add_images('val/degraded', degraded[:4], epoch)
            self.writer.add_images('val/restored', torch.clamp(restored[:4], 0, 1), epoch)
            self.writer.add_images('val/clean', clean[:4], epoch)
        
        return losses.avg, psnrs.avg, ssims.avg

    def train(self):
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            epoch_start_time = time.time()
            
            # Train
            train_loss, train_psnr = self.train_epoch(epoch)
            
            # Validate
            if (epoch + 1) % self.config['training']['val_interval'] == 0:
                val_loss, val_psnr, val_ssim = self.validate(epoch)
                
                print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
                print(f"Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}")
                print(f"Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
                
                # Save best model
                if val_psnr > self.best_psnr:
                    self.best_psnr = val_psnr
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"Best model saved! PSNR: {val_psnr:.2f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch time: {epoch_time:.2f}s\n")
        
        print("Training completed!")
        self.writer.close()

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        # Save regular checkpoint
        save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, save_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_psnr = checkpoint['best_psnr']
        
        print(f"Resumed from epoch {self.start_epoch}, best PSNR: {self.best_psnr:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Train NAFNet-RR for Image Restoration')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.resume:
        config['resume'] = args.resume
    
    # Set random seed
    torch.manual_seed(config.get('seed', 42))
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
