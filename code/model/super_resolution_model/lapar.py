"""from https://github.com/Paper99/SRFBN_CVPR19"""

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from .utils import register_model
from easydict import EasyDict


@register_model
def LAPAR(scale=4, type='C', **kwargs):
    import os
    kernel = os.path.join(os.path.dirname(__file__), 'kernel_72_k5.pkl')
    MODEL = EasyDict()
    if type == 'C':
        MODEL.SCALE = scale
        MODEL.KERNEL_SIZE = 5
        MODEL.KERNEL_PATH = kernel
        MODEL.IN_CHANNEL = 3
        MODEL.N_CHANNEL = 16
        MODEL.RES_BLOCK = 2
        MODEL.N_WEIGHT = 72
    elif type == 'B':
        MODEL.SCALE = scale
        MODEL.KERNEL_SIZE = 5
        MODEL.KERNEL_PATH = kernel
        MODEL.IN_CHANNEL = 3
        MODEL.N_CHANNEL = 24
        MODEL.RES_BLOCK = 3
        MODEL.N_WEIGHT = 72
    elif type == 'A':
        MODEL.SCALE = scale
        MODEL.KERNEL_SIZE = 5
        MODEL.KERNEL_PATH = kernel
        MODEL.IN_CHANNEL = 3
        MODEL.N_CHANNEL = 32
        MODEL.RES_BLOCK = 4
        MODEL.N_WEIGHT = 72
    else:
        raise KeyError()

    model = Network(MODEL)
    return model


class ComponentDecConv(nn.Module):
    def __init__(self, k_path, k_size):
        super(ComponentDecConv, self).__init__()

        kernel = pickle.load(open(k_path, 'rb'))
        kernel = torch.from_numpy(kernel).float().view(-1, 1, k_size, k_size)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        out = F.conv2d(x, weight=self.weight, bias=None, stride=1, padding=0, groups=1)

        return out


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()

        self.k_size = config.KERNEL_SIZE
        self.s = config.SCALE

        self.w_conv = WeightNet(config)
        self.decom_conv = ComponentDecConv(config.KERNEL_PATH, self.k_size)

        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, x, gt=None):
        B, C, H, W = x.size()

        bic = F.interpolate(x, scale_factor=self.s, mode='bicubic', align_corners=False)
        pad = self.k_size // 2
        x_pad = F.pad(bic, pad=(pad, pad, pad, pad), mode='reflect')
        pad_H, pad_W = x_pad.size()[2:]
        x_pad = x_pad.view(B * 3, 1, pad_H, pad_W)
        x_com = self.decom_conv(x_pad).view(B, 3, -1, self.s * H, self.s * W)  # B, 3, N_K, Hs, Ws

        weight = self.w_conv(x)
        weight = weight.view(B, 1, -1, self.s * H, self.s * W)  # B, 1, N_K, Hs, Ws

        out = torch.sum(weight * x_com, dim=2)

        if gt is not None:
            loss_dict = dict(L1=self.criterion(out, gt))
            return loss_dict
        else:
            return out


class Scale(nn.Module):
    def __init__(self, init_value=1e-3):
        super(Scale, self).__init__()
        self.scale = Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class AWRU(nn.Module):
    def __init__(self, nf, kernel_size, wn, act=nn.ReLU(True)):
        super(AWRU, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

        self.body = nn.Sequential(
            wn(nn.Conv2d(nf, nf, kernel_size, padding=kernel_size // 2)),
            act,
            wn(nn.Conv2d(nf, nf, kernel_size, padding=kernel_size // 2)),
        )

    def forward(self, x):
        res = self.res_scale(self.body(x)) + self.x_scale(x)
        return res


class AWMS(nn.Module):
    def __init__(self, nf, out_chl, wn, act=nn.ReLU(True)):
        super(AWMS, self).__init__()
        self.tail_k3 = wn(nn.Conv2d(nf, nf, 3, padding=3 // 2, dilation=1))
        self.tail_k5 = wn(nn.Conv2d(nf, nf, 5, padding=5 // 2, dilation=1))
        self.scale_k3 = Scale(0.5)
        self.scale_k5 = Scale(0.5)
        self.fuse = wn(nn.Conv2d(nf, nf, 3, padding=3 // 2))
        self.act = act
        self.w_conv = wn(nn.Conv2d(nf, out_chl, 3, padding=3 // 2))

    def forward(self, x):
        x0 = self.scale_k3(self.tail_k3(x))
        x1 = self.scale_k5(self.tail_k5(x))
        cur_x = x0 + x1

        fuse_x = self.act(self.fuse(cur_x))
        out = self.w_conv(fuse_x)

        return out


class LFB(nn.Module):
    def __init__(self, nf, wn, act=nn.ReLU(inplace=True)):
        super(LFB, self).__init__()
        self.b0 = AWRU(nf, 3, wn=wn, act=act)
        self.b1 = AWRU(nf, 3, wn=wn, act=act)
        self.b2 = AWRU(nf, 3, wn=wn, act=act)
        self.b3 = AWRU(nf, 3, wn=wn, act=act)
        self.reduction = wn(nn.Conv2d(nf * 4, nf, 3, padding=3 // 2))
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)

    def forward(self, x):
        x0 = self.b0(x)
        x1 = self.b1(x0)
        x2 = self.b2(x1)
        x3 = self.b3(x2)
        res = self.reduction(torch.cat([x0, x1, x2, x3], dim=1))

        return self.res_scale(res) + self.x_scale(x)


class WeightNet(nn.Module):
    def __init__(self, config):
        super(WeightNet, self).__init__()

        in_chl = config.IN_CHANNEL
        nf = config.N_CHANNEL
        n_block = config.RES_BLOCK
        out_chl = config.N_WEIGHT
        scale = config.SCALE

        act = nn.ReLU(inplace=True)
        wn = lambda x: nn.utils.weight_norm(x)

        rgb_mean = torch.FloatTensor([0.4488, 0.4371, 0.4040]).view([1, 3, 1, 1])
        self.register_buffer('rgb_mean', rgb_mean)

        self.head = nn.Sequential(
            wn(nn.Conv2d(in_chl, nf, 3, padding=3 // 2)),
            act,
        )

        body = []
        for i in range(n_block):
            body.append(LFB(nf, wn=wn, act=act))
        self.body = nn.Sequential(*body)

        self.up = nn.Sequential(
            wn(nn.Conv2d(nf, nf * scale ** 2, 3, padding=3 // 2)),
            act,
            nn.PixelShuffle(upscale_factor=scale)
        )

        self.tail = AWMS(nf, out_chl, wn, act=act)

    def forward(self, x):
        x = x - self.rgb_mean
        x = self.head(x)
        x = self.body(x)
        x = self.up(x)
        out = self.tail(x)

        return out
