# -*- coding:utf-8 -*-
# @Time: 2024/1/3 16:00
# @Author: TaoFei
# @FileName: rm_png.py
# @Software: PyCharm

from PIL import Image
import os

def filter_images(txt_path):
    # 读取txt文件中的图像路径
    with open(txt_path, 'r') as file:
        image_paths = file.readlines()

    # 遍历图像路径，删除长或宽小于256的图像
    filtered_paths = []
    for path in image_paths:
        image_path = path.strip()  # 去除换行符和空格
        if os.path.exists(image_path):
            image = Image.open(image_path)
            width, height = image.size
            if width >= 256 and height >= 256:
                filtered_paths.append(image_path)

    # 将筛选后的路径写回txt文件
    with open(txt_path, 'w') as file:
        file.write('\n'.join(filtered_paths))

# 替换 'train.txt' 为你的实际文件路径
filter_images('train.txt')
