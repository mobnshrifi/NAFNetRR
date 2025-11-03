# راهنمای رفع مشکلات و نکات کاربردی

## 🐛 مشکلات رایج و راه‌حل‌ها

### 1. خطای Out of Memory (OOM)

**علائم**:
```
RuntimeError: CUDA out of memory
```

**راه‌حل‌ها**:

```yaml
# در config.yaml
training:
  batch_size: 4  # کاهش batch size (به جای 8 یا 16)
  
data:
  patch_size: 128  # کاهش اندازه patch (به جای 256)
```

```yaml
# استفاده از مدل کوچکتر
model:
  width: 32  # به جای 64
  middle_blk_num: 8  # به جای 12
  reasoning_iterations: 2  # به جای 3
```

**نکات اضافی**:
- از Gradient Accumulation استفاده کنید
- Mixed Precision Training را فعال کنید
- تعداد workers را کم کنید

### 2. Training بسیار کند است

**علل احتمالی**:
1. CPU bottleneck در data loading
2. تنظیمات نامناسب GPU
3. تعداد زیاد workers

**راه‌حل‌ها**:

```yaml
training:
  num_workers: 4  # کاهش workers (اگر CPU ضعیف است)
  pin_memory: true  # فعال‌سازی pin memory
```

```python
# استفاده از Mixed Precision
# در train.py
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# در training loop
with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 3. مدل Overfit می‌شود

**علائم**:
- Training loss کاهش می‌یابد ولی validation loss افزایش می‌یابد
- تفاوت زیاد بین train و val metrics

**راه‌حل‌ها**:

```yaml
# افزایش regularization
training:
  weight_decay: 0.001  # افزایش (از 0.0001)
  
# افزایش data augmentation
data:
  augment: true
  
# استفاده از early stopping
```

```python
# در train.py
from utils import EarlyStopping

early_stopping = EarlyStopping(patience=20)

if early_stopping(val_psnr):
    print("Early stopping triggered")
    break
```

### 4. کیفیت خروجی پایین است

**بررسی‌های اولیه**:

1. **داده‌های آموزش**:
   - تنوع کافی دارند؟
   - Paired به درستی هستند؟
   - Quality خوبی دارند؟

2. **تنظیمات مدل**:
```yaml
model:
  width: 64  # حداقل 64 برای نتایج خوب
  middle_blk_num: 12  # حداقل 12
  reasoning_iterations: 3  # حداقل 3
```

3. **تنظیمات Training**:
```yaml
training:
  epochs: 500  # آموزش کافی
  lr: 0.0002  # learning rate مناسب
```

### 5. Inference بسیار کند است

**بهینه‌سازی‌های ممکن**:

```python
# 1. استفاده از TorchScript
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 2. استفاده از half precision
model.half()
input_tensor = input_tensor.half()

# 3. کاهش iterations
output = model(input, reasoning_iters=2)  # به جای 3
```

```python
# 4. Batch processing
# به جای پردازش تک‌تک، از batch استفاده کنید
batch_input = torch.cat([img1, img2, img3, img4], dim=0)
batch_output = model(batch_input)
```

### 6. خطای در بارگذاری Checkpoint

**خطا**:
```
KeyError: 'model_state_dict'
RuntimeError: Error(s) in loading state_dict
```

**راه‌حل**:

```python
# بررسی محتویات checkpoint
checkpoint = torch.load('model.pth', map_location='cpu')
print(checkpoint.keys())

# اگر ساختار متفاوت است
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
elif 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
```

### 7. نتایج Inconsistent در هر اجرا

**علت**: Random seed تنظیم نشده

**راه‌حل**:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

## 🔍 Debugging Tips

### 1. بررسی Data Loading

```python
# تست dataloader
from dataset import ImageRestorationDataset
from torch.utils.data import DataLoader

dataset = ImageRestorationDataset(...)
loader = DataLoader(dataset, batch_size=1)

# بررسی یک sample
for degraded, clean in loader:
    print(f"Degraded: {degraded.shape}, range: [{degraded.min():.3f}, {degraded.max():.3f}]")
    print(f"Clean: {clean.shape}, range: [{clean.min():.3f}, {clean.max():.3f}]")
    
    # ذخیره برای بررسی بصری
    save_image(degraded[0], 'debug_degraded.png')
    save_image(clean[0], 'debug_clean.png')
    break
```

### 2. بررسی Forward Pass

```python
# تست مدل با ورودی ساده
model = NAFNetRR(...)
dummy_input = torch.randn(1, 3, 256, 256)

print("Testing forward pass...")
try:
    output = model(dummy_input)
    print(f"✓ Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# بررسی gradient flow
output.mean().backward()
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"⚠ No gradient: {name}")
```

### 3. مانیتورینگ Memory Usage

```python
import torch

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

# در training loop
print_memory_usage()
```

### 4. Visualization فرآیند Reasoning

```python
from utils import visualize_reasoning_process

# Visualize کردن تاثیر iterations مختلف
visualize_reasoning_process(
    model,
    test_image,
    save_dir='./debug_visualization',
    num_iterations=5
)
```

## ⚡ بهینه‌سازی عملکرد

### 1. Data Loading Optimization

```python
# استفاده از persistent workers
loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,  # کاهش overhead
    prefetch_factor=2
)
```

### 2. Mixed Precision Training

```python
# نصب
pip install torch>=1.6.0

# استفاده
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data in loader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. Gradient Accumulation

```python
# برای شبیه‌سازی batch size بزرگتر
accumulation_steps = 4

for i, (input, target) in enumerate(loader):
    output = model(input)
    loss = criterion(output, target)
    loss = loss / accumulation_steps
    
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. Model Optimization برای Deployment

```python
# 1. TorchScript
model.eval()
example_input = torch.randn(1, 3, 256, 256)
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# 2. ONNX Export
torch.onnx.export(
    model,
    example_input,
    "model.onnx",
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# 3. Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## 📊 بهترین روش‌ها (Best Practices)

### 1. ساختار پروژه

```
✓ استفاده از Git برای version control
✓ ذخیره منظم checkpoint ها
✓ نگهداری log های دقیق
✓ مستندسازی تغییرات
```

### 2. آموزش

```yaml
# تنظیمات توصیه شده
✓ از cosine scheduler استفاده کنید
✓ validation منظم انجام دهید
✓ early stopping پیاده‌سازی کنید
✓ چند مدل با seed های مختلف train کنید
```

### 3. ارزیابی

```python
# ارزیابی جامع
✓ روی چند دیتاست تست کنید
✓ metrics مختلف محاسبه کنید (PSNR, SSIM, LPIPS)
✓ visual quality را چک کنید
✓ inference speed را اندازه‌گیری کنید
```

## 🎯 نکات تخصصی

### برای GoPro Deblurring

```yaml
model:
  width: 64
  middle_blk_num: 12
  reasoning_iterations: 3

data:
  patch_size: 256

training:
  batch_size: 8
  epochs: 3000
  lr: 0.0002
```

### برای SIDD Denoising

```yaml
model:
  width: 32
  middle_blk_num: 8
  reasoning_iterations: 4

data:
  patch_size: 128

training:
  batch_size: 16
  epochs: 2000
  lr: 0.0003
```

### برای Real-time Applications

```yaml
model:
  width: 32
  middle_blk_num: 4
  enc_blk_nums: [1, 1, 1, 2]
  dec_blk_nums: [1, 1, 1, 1]
  reasoning_iterations: 2
  reasoning_positions: ['middle']  # فقط middle

training:
  batch_size: 32
```

## 🆘 دریافت کمک

اگر مشکل شما حل نشد:

1. **GitHub Issues**: مشکل خود را با جزئیات کامل گزارش دهید
2. **Discussions**: در بخش discussions سوال بپرسید
3. **Email**: برای مسائل خاص ایمیل بزنید

### اطلاعات مورد نیاز برای گزارش مشکل

```
- نسخه Python و PyTorch
- مشخصات GPU
- فایل config.yaml
- پیام خطای کامل
- کد مربوطه
- مراحل بازتولید مشکل
```

---

**یادآوری**: همیشه ابتدا با مدل و دیتاست کوچک تست کنید!
