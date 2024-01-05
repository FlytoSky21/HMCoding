# -*- coding:utf-8 -*-
# @Time: 2023/11/19 18:02
# @Author: TaoFei
# @FileName: dcc2023_sft.py
# @Software: PyCharm

from compressai.models import CompressionModel
import math
import warnings

import torch
import torch.nn as nn
import re
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
# from models.yolov3_models import load_model
from pytorchyolo import detect, my_models

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def load_pretrained_model(model, pretrained_state_dict, prefix_length):
    model_state_dict = model.state_dict()

    # 去掉前缀并加载参数
    for pretrain_key, pretrain_value in pretrained_state_dict.items():
        if len(pretrain_key) > prefix_length:
            # 去掉前缀
            model_key = pretrain_key[prefix_length:]
            if model_key in model_state_dict:
                model_state_dict[model_key] = pretrain_value
    model.load_state_dict(model_state_dict)


def freeze_para(model):
    for param in model.parameters():
        param.requires_grad = False


class DCC2023Model(CompressionModel):
    def __init__(self, Cs=256, N=192, M1=64, M2=128, M=192):
        super(DCC2023Model, self).__init__(entropy_bottleneck_channels=N)
        yolov3 = my_models.load_model("/home/adminroot/taofei/DCC2023fuxian/config/yolov3.cfg",
                                      "/home/adminroot/taofei/DCC2023fuxian/config/yolov3.weights")
        self.yolov3_front = my_models.load_front_model("/home/adminroot/taofei/DCC2023fuxian/config/yolov3_front.cfg")
        self.yolov3_back = my_models.load_back_model("/home/adminroot/taofei/DCC2023fuxian/config/yolov3_back.cfg")

        yolov3_dict = yolov3.state_dict()
        yolov3_front_dict = self.yolov3_front.state_dict()
        yolov3_front_pretrained = {k: v for k, v in yolov3_dict.items() if k in yolov3_front_dict}
        yolov3_front_dict.update(yolov3_front_pretrained)
        self.yolov3_front.load_state_dict(yolov3_front_dict)

        yolov3_back_dict = self.yolov3_back.state_dict()
        new_yolov3_back_dict = {}
        for k, v in yolov3_dict.items():
            # 将键中的数字-13
            k = re.sub(r'(\d+)', lambda x: str(int(x.group(1)) - 13), k)
            if k in yolov3_back_dict:
                new_yolov3_back_dict[k] = v

        yolov3_back_dict.update(new_yolov3_back_dict)
        self.yolov3_back.load_state_dict(yolov3_back_dict)

        # self.yolov3_front = torch.nn.Sequential(*list(self.yolov3.module_list)[:13])
        # self.yolov3_back = torch.nn.Sequential(*list(self.yolov3.module_list)[13:])

        # Freeze the parameters of yolo_front and yolo_back
        for param in self.yolov3_front.parameters():
            param.requires_grad = False
        for param in self.yolov3_back.parameters():
            param.requires_grad = False

        self.modules0 = nn.Sequential()
        self.modules0.add_module("conv_0",
                                 nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.modules0.add_module(f"batch_norm_0", nn.BatchNorm2d(32, momentum=0.1, eps=1e-5))
        self.modules0.add_module(f"leaky_0", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules0, yolov3_front_pretrained, 14)
        freeze_para(self.modules0)

        self.modules1 = nn.Sequential()
        self.modules1.add_module("conv_1",
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,
                                           bias=False))
        self.modules1.add_module(f"batch_norm_1", nn.BatchNorm2d(64, momentum=0.1, eps=1e-5))
        self.modules1.add_module(f"leaky_1", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules1, yolov3_front_pretrained, 14)
        freeze_para(self.modules1)

        self.modules2 = nn.Sequential()
        self.modules2.add_module("conv_2",
                                 nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, bias=False))
        self.modules2.add_module(f"batch_norm_2", nn.BatchNorm2d(32, momentum=0.1, eps=1e-5))
        self.modules2.add_module(f"leaky_2", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules2, yolov3_front_pretrained, 14)
        freeze_para(self.modules2)

        self.modules3 = nn.Sequential()
        self.modules3.add_module("conv_3",
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.modules3.add_module(f"batch_norm_3", nn.BatchNorm2d(64, momentum=0.1, eps=1e-5))
        self.modules3.add_module(f"leaky_3", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules3, yolov3_front_pretrained, 14)
        freeze_para(self.modules3)

        # self.modules4 = nn.Sequential()
        # self.modules4.add_module(f"shortcut_4", nn.Sequential())

        self.modules5 = nn.Sequential()
        self.modules5.add_module("conv_5",
                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
                                           bias=False))
        self.modules5.add_module(f"batch_norm_5", nn.BatchNorm2d(128, momentum=0.1, eps=1e-5))
        self.modules5.add_module(f"leaky_5", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules5, yolov3_front_pretrained, 14)
        freeze_para(self.modules5)

        self.modules6 = nn.Sequential()
        self.modules6.add_module("conv_6",
                                 nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias=False))
        self.modules6.add_module(f"batch_norm_6", nn.BatchNorm2d(64, momentum=0.1, eps=1e-5))
        self.modules6.add_module(f"leaky_6", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules6, yolov3_front_pretrained, 14)
        freeze_para(self.modules6)

        self.modules7 = nn.Sequential()
        self.modules7.add_module("conv_7",
                                 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                                           bias=False))
        self.modules7.add_module(f"batch_norm_7", nn.BatchNorm2d(128, momentum=0.1, eps=1e-5))
        self.modules7.add_module(f"leaky_7", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules7, yolov3_front_pretrained, 14)
        freeze_para(self.modules7)

        # self.modules8 = nn.Sequential()
        # self.modules8.add_module(f"shortcut_8", nn.Sequential())

        self.modules9 = nn.Sequential()
        self.modules9.add_module("conv_9",
                                 nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, bias=False))
        self.modules9.add_module(f"batch_norm_9", nn.BatchNorm2d(64, momentum=0.1, eps=1e-5))
        self.modules9.add_module(f"leaky_9", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules9, yolov3_front_pretrained, 14)
        freeze_para(self.modules9)

        self.modules10 = nn.Sequential()
        self.modules10.add_module("conv_10",
                                  nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                                            bias=False))
        self.modules10.add_module(f"batch_norm_10", nn.BatchNorm2d(128, momentum=0.1, eps=1e-5))
        self.modules10.add_module(f"leaky_10", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules10, yolov3_front_pretrained, 15)
        freeze_para(self.modules10)

        # self.modules11 = nn.Sequential()
        # self.modules11.add_module(f"shortcut_11", nn.Sequential())

        self.modules12 = nn.Sequential()
        self.modules12.add_module("conv_12",
                                  nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,
                                            bias=False))
        self.modules12.add_module(f"batch_norm_12", nn.BatchNorm2d(256, momentum=0.1, eps=1e-5))
        self.modules12.add_module(f"leaky_12", nn.LeakyReLU(0.1))
        load_pretrained_model(self.modules12, yolov3_front_pretrained, 15)
        freeze_para(self.modules12)

        self.gs_a = nn.Sequential(
            conv(Cs + 3, N, 5, 1),
            GDN(N),
            conv(N, N, 5, 1),
            GDN(N),
            conv(N, M1, 5, 2),
        )
        self.gs_s = nn.Sequential(
            deconv(M1, N, 5, 1),
            GDN(N, inverse=True),
            deconv(N, N, 5, 1),
            GDN(N, inverse=True),
            deconv(N, Cs, 5, 2),
        )
        self.hs_a = nn.Sequential(
            conv(M1, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.hs_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M1, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )
        self.gaussian_conditional = GaussianConditional(None)
        self.Cs = Cs
        self.N = N
        self.M1 = M1
        self.M2 = M2
        self.M = M

    def forward(self, x, x_lr):
        # baselayer
        img_size = x.size(2)
        f0 = self.modules0(x)  # 32*256*256
        f1 = self.modules1(f0)  # 64*128*128
        f2 = self.modules2(f1)  # 32*128*128
        f3 = self.modules3(f2)  # 64*128*128
        f4 = f3 + f1  # 64*128*128
        f5 = self.modules5(f4)  # 128*64*64    **
        f6 = self.modules6(f5)  # 64*64*64
        f7 = self.modules7(f6)  # 128*64*64
        f8 = f7 + f5  # 128*64*64
        f9 = self.modules9(f8)  # 64*64*64
        f10 = self.modules10(f9)  # 128*64*64
        f11 = f10 + f8  # 128*64*64
        s = self.modules12(f11)  # 256*32*32   **

        # s = self.yolov3_front(x)
        f = torch.cat([s, x_lr], dim=1)
        y1 = self.gs_a(f)
        z1 = self.hs_a(torch.abs(y1))
        z1_hat, z1_likelihoods = self.entropy_bottleneck(z1)
        scales_hat_z1 = self.hs_s(z1_hat)
        y1_hat, y1_likelihoods = self.gaussian_conditional(y1, scales_hat_z1)
        s_hat = self.gs_s(y1_hat)
        # t_hat = self.yolov3_back(s_hat, img_size)

        return {
            "base_likelihoods": {"y1": y1_likelihoods, "z1": z1_likelihoods},
            "s": s,
            "s_hat": s_hat,
            # "t_hat":t_hat
        }
