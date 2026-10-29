import numpy as np
import torch
from torch import nn
from torch.nn import init



# class SEAttention(nn.Module):

#     def __init__(self, in_channels=512,reduction=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // reduction, in_channels, bias=False),
#             nn.Sigmoid()
#         )


#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
class SEAttention(nn.Module):
    def __init__(self, reduction=16):
        super().__init__()
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = None
        self.initialized_channels = None

    def _init_fc(self, channels):
        """根据通道数初始化全连接层"""
        reduced_channels = max(channels // self.reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
        self.initialized_channels = channels

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 如果是第一次前向传播或者通道数变化，初始化fc模块
        if self.fc is None or c != self.initialized_channels:
            self._init_fc(c)
            # 将模块移动到正确的设备
            self.fc = self.fc.to(x.device)
        
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    se = SEAttention()
    output=se(input)
    print(output.shape)
