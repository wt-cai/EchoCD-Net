
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import math
class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             kernel_size=3, padding=1, 
                             groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class LoRALayer(nn.Module):
    def __init__(self, dim, r=8):
        super().__init__()
        self.r = r

        self.lora_A = nn.Linear(dim, r, bias=False)
        self.lora_B = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))  # ΔW = BA

class EfficientSE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class BAapter(nn.Module):
    def __init__(self, in_channels, reduction=4, K=3, r=8):
        super().__init__()
        self.K = K
        mid_channels = max(in_channels // 2, 32)
        
        self.proj_down = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.LayerNorm(mid_channels),
            nn.GELU()
        )

        self.offset_conv = CBR(mid_channels, 2*K*K)  
        self.deform_conv = DeformConv2d(mid_channels, mid_channels, 
                                       kernel_size=K, padding=K//2,
                                       groups=mid_channels) 
        
        self.patch_conv = CBR(mid_channels, mid_channels)
        self.unfold = nn.Unfold(kernel_size=K, padding=K//2)
        
        self.fg_proj = nn.Sequential(
            CBR(mid_channels, mid_channels//2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_channels//2, K*K*K*K)
        )
        self.bg_proj = nn.Sequential(
            CBR(mid_channels, mid_channels//2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_channels//2, K*K*K*K)
        )
        self.se = EfficientSE(mid_channels, reduction)
        self.lora = LoRALayer(mid_channels, r)

        self.proj_up = nn.Linear(mid_channels, in_channels)

    def forward(self, x):
        B, H, W, C = x.shape
        x_down = self.proj_down(x)  # (B, H, W, mid_C)
        x_conv = x_down.permute(0, 3, 1, 2)  # (B, mid_C, H, W)
        offset = self.offset_conv(x_conv)
        deform_feat = self.deform_conv(x_conv, offset)
        patches = self.unfold(self.patch_conv(x_conv))  # (B, mid_C*K*K, H*W)
        patches = patches.view(B, -1, self.K*self.K, H, W)  # (B, mid_C, K*K, H, W)
        fg_weight = F.softmax(self.fg_proj(x_conv).view(B, self.K*self.K, self.K*self.K), dim=-1)
        bg_weight = F.softmax(self.bg_proj(x_conv).view(B, self.K*self.K, self.K*self.K), dim=-1)
        weights = (fg_weight + bg_weight).view(B, 1, self.K*self.K, self.K*self.K, 1)
        attended = torch.einsum('bckhw,bnklm->bchw', patches, weights)
        merged = deform_feat + attended
        merged = self.se(merged)
        merged = merged.permute(0, 2, 3, 1)  # (B, H, W, mid_C)
        lora_adjusted = self.lora(merged) + merged
        out = self.proj_up(F.gelu(lora_adjusted))
        return out



def main():
    adapter = BAapter(in_channels=768)
    x = torch.randn(10, 64, 64, 768) 
    y = adapter(x)
    print(y.shape) 


if __name__ == "__main__":
    main()