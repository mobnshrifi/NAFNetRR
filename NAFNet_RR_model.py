"""
NAFNet with Recurrent Reasoning for Image Restoration
معماری NAFNet با قابلیت Reasoning بازگشتی در فضای نهان
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    """Layer Normalization سفارشی برای بهبود عملکرد"""
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(dim=0), None


class LayerNorm2d(nn.Module):
    """Layer Normalization برای تصاویر 2D"""
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    """Simple Gate Mechanism - بدون استفاده از activation functions"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    بلوک اصلی NAFNet
    Simple Baseline بدون Nonlinear Activation Functions
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        )

        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return y + x * self.gamma


class RecurrentReasoningModule(nn.Module):
    """
    ماژول Recurrent Reasoning برای بهبود تدریجی در فضای نهان
    این ماژول feature maps را چندین بار refine می‌کند
    """
    def __init__(self, channels, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Knowledge Base: ذخیره اطلاعات از iteration قبلی
        self.knowledge_conv = nn.Conv2d(channels, channels, 3, 1, 1)
        
        # Reasoning Unit: پردازش و refine کردن features
        self.reasoning_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, 1, 1),  # Concat: current + knowledge
                LayerNorm2d(channels),
                nn.Conv2d(channels, channels, 3, 1, 1),
                SimpleGate() if i % 2 == 0 else nn.Identity()
            ) for i in range(num_iterations)
        ])
        
        # Update Gate: تصمیم‌گیری برای به‌روزرسانی
        self.update_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),
                nn.Sigmoid()
            ) for _ in range(num_iterations)
        ])
        
        # Residual weights
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(num_iterations)
        ])

    def forward(self, x, num_iters=None):
        """
        Args:
            x: input features [B, C, H, W]
            num_iters: تعداد iterations (اگر None باشد، از self.num_iterations استفاده می‌شود)
        Returns:
            refined features after recurrent reasoning
        """
        if num_iters is None:
            num_iters = self.num_iterations
            
        # Initialize knowledge base
        knowledge = torch.zeros_like(x)
        current = x
        
        # Recurrent reasoning iterations
        for i in range(num_iters):
            # Update knowledge base
            knowledge = self.knowledge_conv(current)
            
            # Concat current features with knowledge
            combined = torch.cat([current, knowledge], dim=1)
            
            # Reasoning step
            reasoned = self.reasoning_blocks[i](combined)
            
            # Update gate (چقدر از reasoning جدید استفاده کنیم)
            gate = self.update_gates[i](combined)
            
            # Update current features with gated reasoning
            current = current * (1 - gate) + reasoned * gate
            
            # Residual connection با وزن قابل یادگیری
            current = current * self.alphas[i] + x * (1 - self.alphas[i])
            
        return current


class NAFNetRR(nn.Module):
    """
    NAFNet with Recurrent Reasoning
    مدل اصلی که NAFNet را با Recurrent Reasoning ترکیب می‌کند
    """
    def __init__(self, img_channel=3, width=64, middle_blk_num=12, 
                 enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2],
                 reasoning_iterations=3, reasoning_positions=['middle', 'decoder']):
        """
        Args:
            img_channel: تعداد کانال‌های ورودی
            width: تعداد کانال‌های base
            middle_blk_num: تعداد بلوک‌های middle
            enc_blk_nums: تعداد بلوک‌ها در هر stage encoder
            dec_blk_nums: تعداد بلوک‌ها در هر stage decoder
            reasoning_iterations: تعداد iterations در Recurrent Reasoning
            reasoning_positions: کجاها از Recurrent Reasoning استفاده شود
                                ['middle', 'decoder', 'bottleneck']
        """
        super().__init__()
        
        self.reasoning_iterations = reasoning_iterations
        self.reasoning_positions = reasoning_positions
        
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, 
                               kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, 
                                kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])
        
        # Recurrent Reasoning Module در Middle
        if 'middle' in reasoning_positions:
            self.middle_reasoning = RecurrentReasoningModule(chan, reasoning_iterations)
        
        # Recurrent Reasoning Modules در Decoder stages
        self.decoder_reasoning_modules = nn.ModuleList()
        
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            
            self.decoders.append(
                nn.Sequential(*[NAFBlock(chan) for _ in range(num)])
            )
            
            # اضافه کردن Reasoning به هر stage decoder
            if 'decoder' in reasoning_positions:
                self.decoder_reasoning_modules.append(
                    RecurrentReasoningModule(chan, reasoning_iterations)
                )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, reasoning_iters=None):
        """
        Args:
            inp: input image [B, C, H, W]
            reasoning_iters: تعداد iterations برای reasoning (None = استفاده از default)
        """
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        
        # Encoder
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Middle blocks
        x = self.middle_blks(x)
        
        # Recurrent Reasoning در Middle
        if 'middle' in self.reasoning_positions:
            x = self.middle_reasoning(x, reasoning_iters)

        # Decoder
        decoder_reasoning_idx = 0
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
            
            # Recurrent Reasoning در Decoder
            if 'decoder' in self.reasoning_positions:
                x = self.decoder_reasoning_modules[decoder_reasoning_idx](x, reasoning_iters)
                decoder_reasoning_idx += 1

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """Pad image to be divisible by padder_size"""
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x


def count_parameters(model):
    """محاسبه تعداد پارامترهای قابل آموزش"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # تست مدل
    print("Testing NAFNet with Recurrent Reasoning...")
    
    # مدل با reasoning در middle و decoder
    model = NAFNetRR(
        img_channel=3,
        width=32,
        middle_blk_num=4,
        enc_blk_nums=[1, 1, 1, 2],
        dec_blk_nums=[1, 1, 1, 1],
        reasoning_iterations=3,
        reasoning_positions=['middle', 'decoder']
    )
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # تست forward pass
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        # استفاده از تعداد iteration های مختلف
        for iters in [1, 2, 3]:
            out = model(x, reasoning_iters=iters)
            print(f"Input shape: {x.shape}, Output shape (iters={iters}): {out.shape}")
    
    print("\nModel architecture created successfully!")
