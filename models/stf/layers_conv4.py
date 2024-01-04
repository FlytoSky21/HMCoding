import torch.nn as nn
import torch.nn.functional as F


class SFTLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv2 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv3 = nn.Conv2d(in_ch, out_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv2 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv3 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale0 = F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True)
        scale1 = F.leaky_relu(self.SFT_scale_conv1(scale0), 0.1, inplace=True)
        scale = self.SFT_scale_conv3(F.leaky_relu(self.SFT_scale_conv2(scale1), 0.1, inplace=True))
        shift0 = F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True)
        shift2 = F.leaky_relu(self.SFT_shift_conv1(shift0), 0.1, inplace=True)
        shift = self.SFT_shift_conv3(F.leaky_relu(self.SFT_shift_conv2(shift2), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class SFTLayerNoBias(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SFTLayerNoBias, self).__init__()
        # self.SFT_scale_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        # self.SFT_scale_conv1 = nn.Conv2d(in_ch, out_ch, 1)
        # self.SFT_shift_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        # self.SFT_shift_conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.SFT_scale_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv2 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_scale_conv3 = nn.Conv2d(in_ch, out_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv2 = nn.Conv2d(in_ch, in_ch, 1)
        self.SFT_shift_conv3 = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        # scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True))
        scale0 = F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.1, inplace=True)
        scale1 = F.leaky_relu(self.SFT_scale_conv1(scale0), 0.1, inplace=True)
        scale = self.SFT_scale_conv3(F.leaky_relu(self.SFT_scale_conv2(scale1), 0.1, inplace=True))
        # shift0 = F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.1, inplace=True)
        # shift2 = F.leaky_relu(self.SFT_shift_conv1(shift0), 0.1, inplace=True)
        # shift = self.SFT_shift_conv3(F.leaky_relu(self.SFT_shift_conv2(shift2), 0.1, inplace=True))
        return x[0] * (scale + 1)
