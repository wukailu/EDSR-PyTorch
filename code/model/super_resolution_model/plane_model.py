import torch
import torch.nn as nn
import torch.nn.functional as F

from .SPADE_Norm import SPADE
from .utils import register_model

@register_model
def Plane(**kwargs):
    model = Plane_Model(**kwargs)
    return model


class Plane_Model(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, scale=4, **kwargs):
        super(Plane_Model, self).__init__()
        self.num_modules = num_modules
        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)
        self.out_nc = out_nc
        self.scale = scale

        self.features = nn.ModuleList([RepBlock(in_channel=nf, **kwargs) for i in range(num_modules)])

        self.up_conv = conv_layer(nf, out_nc * (scale ** 2), 1, 1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, input):
        out = self.fea_conv(input)
        for m in self.features:
            if isinstance(m, RepBlock):
                out = m(out, input)
            else:
                out = m(out)
        out = self.up_conv(out) + torch.cat([input] * (self.scale * self.scale), dim=1)
        return self.shuffle(out)


def conv_layer(in_channels, out_channels, kernel_size, stride=1, bias=True):
    padding = int((kernel_size - 1) / 2)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ESA(nn.Module):
    total_time = 0

    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m


class RepBlock(nn.Module):
    def __init__(self, in_channel, norm_type='spade', conv_in_block=1, use_act=True, norm_before_relu=False,
                 use_esa=False, use_spade=True, large_ori=False, **kwargs):
        super().__init__()
        self.norm_type = norm_type
        self.large_ori = large_ori

        conv_in = in_channel + (3 * (in_channel // 3) if large_ori else 3)
        self.convs = nn.ModuleList()
        for i in range(conv_in_block):
            if norm_before_relu:
                if norm_type == 'spade':
                    self.convs.append(SPADE('spadebatch3x3', in_channel))
                elif norm_type == 'bn':
                    self.norm = nn.BatchNorm2d(in_channel)
                elif norm_type == 'in':
                    self.norm = nn.InstanceNorm2d(in_channel)
            if use_act:
                self.convs.append(activation('lrelu', neg_slope=0.05))
            self.convs.append(conv_layer(conv_in, in_channel, 3))

        if use_spade:
            self.convs.append(SPADE('spadebatch3x3', in_channel))

        if use_esa:
            self.convs.append(ESA(in_channel, nn.Conv2d))

    def forward(self, x, ori_input):
        out = x
        if self.large_ori:
            ori_input = torch.cat([ori_input] * (x.size(1) // ori_input.size(1)), dim=1)
        for m in self.convs:
            if isinstance(m, nn.Conv2d):
                out = torch.cat([ori_input, out], dim=1)
            # TODO 如何解决 ori_input 只占3 channel 但 conv parameters 分布均匀的问题
            out = m(out)

        return out

