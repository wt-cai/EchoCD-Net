import numpy as np
import torch
from torchvision import transforms as T
import torchvision.transforms.functional as F
from PIL import Image

class JointTransform3D_Echo:
    def __init__(self,
                 img_size=256,
                 crop=(32, 32),
                 p_flip=0.5,
                 p_rota=0.5,
                 p_scale=0.5,
                 p_gaussn=0.5,
                 p_contr=0.5,
                 color_jitter_params=(0.1, 0.1, 0.1, 0.1),
                 p_random_affine=0.5):
        self.crop = crop
        self.p_flip = p_flip
        self.p_rota = p_rota
        self.p_scale = p_scale
        self.p_gaussn = p_gaussn
        self.p_contr = p_contr
        self.color_jitter_params = color_jitter_params
        self.p_random_affine = p_random_affine
        self.img_size = img_size
        
        if color_jitter_params:
            self.color_tf = T.ColorJitter(*color_jitter_params)

    def __call__(self, image, mask):
        image = image.astype(np.uint8)
        image_list, mask_list = [], []

        for image_ in image:
            image_ = image_.transpose(1, 2, 0)
            image_ = F.to_pil_image(image_)
            image_list.append(image_)

        for mask_ in mask:
            mask_ = F.to_pil_image(mask_)
            mask_list.append(mask_)

        # Apply random crop
        if self.crop:
            i, j, h, w = T.RandomCrop.get_params(image_list[0], self.crop)
            for idx, image_ in enumerate(image_list):
                image_list[idx] = F.crop(image_, i, j, h, w)
            for idx, mask_ in enumerate(mask_list):
                mask_list[idx] = F.crop(mask_, i, j, h, w)

        # Random flip
        if np.random.rand() < self.p_flip:
            image_list = [F.hflip(image_) for image_ in image_list]
            mask_list = [F.hflip(mask_) for mask_ in mask_list]

        # Random rotation
        if np.random.rand() < self.p_rota:
            angle = T.RandomRotation.get_params((-30, 30))
            image_list = [F.rotate(image_, angle) for image_ in image_list]
            mask_list = [F.rotate(mask_, angle) for mask_ in mask_list]

        # Random scale
        if np.random.rand() < self.p_scale:
            scale = np.random.uniform(1, 1.3)
            new_h, new_w = int(self.img_size * scale), int(self.img_size * scale)
            image_list = [F.resize(image_, (new_h, new_w), T.InterpolationMode.BILINEAR) for image_ in image_list]
            mask_list = [F.resize(mask_, (new_h, new_w), T.InterpolationMode.NEAREST) for mask_ in mask_list]
            i, j, h, w = T.RandomCrop.get_params(image_list[0], (self.img_size, self.img_size))
            for idx, image_ in enumerate(image_list):
                image_list[idx] = F.crop(image_, i, j, h, w)
            for idx, mask_ in enumerate(mask_list):
                mask_list[idx] = F.crop(mask_, i, j, h, w)

        # Random contrast
        if np.random.rand() < self.p_contr:
            contr_tf = T.ColorJitter(contrast=(0.8, 2.0))
            image_list = [contr_tf(image_) for image_ in image_list]

        # Apply color jitter (only on image)
        if self.color_jitter_params:
            image_list = [self.color_tf(image_) for image_ in image_list]

        # Random affine transform
        if np.random.rand() < self.p_random_affine:
            affine_params = T.RandomAffine(180).get_params(
                (-90, 90), (1, 1), (2, 2), (-45, 45), self.crop)
            image_list = [F.affine(image_, *affine_params) for image_ in image_list]
            mask_list = [F.affine(mask_, *affine_params) for mask_ in mask_list]

        # Resize all images and masks to the target size
        image_list = [F.resize(image_, (self.img_size, self.img_size), T.InterpolationMode.BILINEAR) for image_ in image_list]
        mask_list = [F.resize(mask_, (self.img_size, self.img_size), T.InterpolationMode.NEAREST) for mask_ in mask_list]

        # Stack image and mask
        image = np.stack(image_list)
        image = image.transpose(0, 3, 1, 2)
        mask = np.stack(mask_list)

        # Convert to tensor
        image = torch.tensor(image)
        mask = torch.tensor(mask)

        # Return the original and augmented images
        augmented_image = torch.cat([image, image], dim=0)
        augmented_mask = torch.cat([mask, mask], dim=0)

        return augmented_image, augmented_mask


import numpy as np
import torch

def main():
    # 假设我们有10个样本的图像和掩码
    image = np.random.rand(10, 256, 256, 3)  # 假设的图像数据，10个256x256的RGB图像
    mask = np.random.randint(0, 2, (10, 256, 256))  # 假设的掩码数据，10个256x256的二值掩码

    # 打印原始数据集的形状
    print("Original image shape:", image.shape)
    print("Original mask shape:", mask.shape)

    # 初始化数据增强器
    augmenter = JointTransform3D_Echo(
        img_size=256, 
        crop=(32, 32),  # 随机裁剪的目标大小
        p_flip=0.5,     # 50%概率水平翻转
        p_rota=0.5,     # 50%概率旋转
        p_scale=0.5,    # 50%概率缩放
        p_contr=0.5,    # 50%概率对比度调整
        color_jitter_params=(0.1, 0.1, 0.1, 0.1),  # 颜色抖动参数
        p_random_affine=0.5  # 50%概率随机仿射变换
    )

    # 调用数据增强方法
    augmented_image, augmented_mask = augmenter(image, mask)

    # 打印增强后数据集的形状
    print("Augmented image shape:", augmented_image.shape)
    print("Augmented mask shape:", augmented_mask.shape)

if __name__ == "__main__":
    main()