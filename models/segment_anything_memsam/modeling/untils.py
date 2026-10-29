from tkinter import X
from unittest import skip
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock, Adapter, AugAdapter
import math
from .rein import LoRAReins
from typing import List
from torch import Tensor


def do_once(func: callable, workname=None):
    global done_work
    if workname is None:
        workname=func()
    if workname not in done_work:
        func()
        done_work.add(workname)

def interpolate_pos_encoding(self, x, w, h):
    previous_dtype = x.dtype
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    if npatch == N:
        # support for w!=h
        self.do_once(
            lambda: print(
                f"h={h},w={w},x.shape={x.shape},pos_embed.shape={self.pos_embed.shape}"
            ),
            "interpolate_pos_encoding",
        )
        return self.pos_embed
    if npatch == N and w == h:
        return self.pos_embed
    pos_embed = self.pos_embed.float()
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = x.shape[-1]
    w0 = w // self.patch_size
    h0 = h // self.patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1

    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(
            1, int(math.sqrt(N)), int(math.sqrt(N)), dim
        ).permute(0, 3, 1, 2),
        scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
        mode="bicubic",
    )

    assert (
        int(w0) == patch_pos_embed.shape[-2]
        and int(h0) == patch_pos_embed.shape[-1]
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(
        previous_dtype
    )

def prepare_tokens_with_masks(self, x, masks=None):
    B, nc, w, h = x.shape
    x = self.patch_embed(x)
    # print(f"After patch_embed: {x.shape}")
    if masks is not None:
        x = torch.where(
            masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x
        )

    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + self.interpolate_pos_encoding(x, w, h)
    return x

# def get_pyramid_feature(
#     features: List[Tensor], scales: List[float] = [4, 2, 1, 0.5]
# ) -> List[Tensor]:
#     """
#     features: List[Tensor]
#     scales: List[float]=[4,2,1,0.5]
#     """
    
#     pyramid_feature = []
#     for i in range(len(features)):
#         pyramid_feature.append(
#             nn.functional.interpolate(
#                 features[i],
#                 scale_factor=scales[i],
#                 mode="bilinear",
#                 align_corners=False,
#             )
#         )
#     return pyramid_feature

def get_pyramid_feature(features: List[Tensor], scales: List[float] = [4, 2, 1, 0.5]) -> List[Tensor]:
    """
    Generate pyramid features from input features with scaling.
    features: List[Tensor], input feature maps.
    scales: List[float], scaling factors for each feature map.
    """
    # 如果 scales 长度小于 features，则动态生成 scales
    if len(scales) != len(features):
        scales = [4.0 / (2 ** i) for i in range(len(features))]

    pyramid_feature = []
    for i in range(len(features)):
        h, w = features[i].shape[-2:]  # 获取原始尺寸
        new_h, new_w = int(h * scales[i]), int(w * scales[i])
        new_h, new_w = max(1, new_h), max(1, new_w)  # 确保输出尺寸至少为 1
        pyramid_feature.append(
            nn.functional.interpolate(
                features[i],
                size=(new_h, new_w),  # 使用目标尺寸
                mode="bilinear",
                align_corners=False,
            )
        )
    return pyramid_feature