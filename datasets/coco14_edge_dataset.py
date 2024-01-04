# -*- coding:utf-8 -*-
# @Time: 2023/11/18 23:16
# @Author: TaoFei
# @FileName: vimeo90k_dataset.py
# @Software: PyCharm


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import ImageFile
from skimage import color, feature
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Vimeo90KDataset(Dataset):
    def __init__(self, root, transform=None, split="train"):
        img_path = Path(root) / f"{split}.txt"
        if not img_path.is_file():
            raise RuntimeError(f"List file not found: {img_path}")
        with open(img_path, "r") as f:
            self.samples = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        # 随机裁剪为 256x256
        img = self.transform(img)

        # 获取降采样后的 32x32 图像
        img_lr = img.resize((256 // 8, 256 // 8), Image.BICUBIC)
        gray_img_lr = color.rgb2gray(img_lr)
        # 使用canny算子进行边缘检测
        edge_img_lr = feature.canny(gray_img_lr, sigma=1)

        # 转换为 PyTorch 的 Tensor
        img = transforms.ToTensor()(img)
        img_lr = transforms.ToTensor()(img_lr)
        edge_img_lr = transforms.ToTensor()(edge_img_lr)

        return img, img_lr, edge_img_lr


if __name__ == "__main__":
    # 示例用法
    data_dir = "/home/adminroot/taofei/dataset/vimeo_septuplet"
    transform = transforms.Compose([transforms.ToTensor()])

    # 创建数据集实例
    vimeo_dataset = Vimeo90KDataset(data_dir, transform=transform)

    # 获取数据集的一个样本
    sample_lr, sample_hr = vimeo_dataset[0]

    plt.imshow(sample_lr)
    plt.show()

    plt.imshow(sample_hr)
    plt.show()

    # 打印样本的形状
    print("Low Resolution Sample Shape:", sample_lr.shape)
    print("High Resolution Sample Shape:", sample_hr.shape)
