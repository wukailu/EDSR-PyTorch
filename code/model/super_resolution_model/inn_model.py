import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.utils import Normalize
from .SPADE_Norm import SPADE
from .utils import register_model, unpack_feature, pack_feature
from torchvision.models import vgg16_bn


# TODO: fix memory bugs
from .. import freeze


@register_model
def INN(**kwargs):
    model = INN_Model(**kwargs)
    return model


class INN_Model(nn.Module):
    def __init__(self, in_nc=3, n_feats=50, num_modules=4, out_nc=3, scale=4, block_skip=True, vgg_feat=False, skip_cons=(1, 1, 1), **kwargs):
        super(INN_Model, self).__init__()
        nf = n_feats
        self.num_modules = num_modules
        self.block_skip = block_skip
        self.vgg_feat = vgg_feat
        self.skip_cons = skip_cons

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        block_type = Rep_RFDB
        self.features = nn.ModuleList()
        for i in range(num_modules):
            self.features.append(block_type(in_channels=nf, vgg_feat=vgg_feat, **kwargs))

        if self.block_skip:
            self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=scale)
        self.scale_idx = 0

        if vgg_feat:
            features = [Normalize(mean=(0.485*255, 0.456*255, 0.406*255), std=(0.229*255, 0.224*255, 0.225*255))]
            features = features + [m for m in vgg16_bn(pretrained=True).features]
            self.vgg = nn.Sequential(*features[:13])
            freeze(self.vgg)

    def forward(self, input):
        if self.vgg_feat:
            feat_vgg = self.vgg(input)
        else:
            feat_vgg = None

        out_fea = self.fea_conv(input)
        records = []
        x = out_fea
        for i in range(self.num_modules):
            x = self.features[i](x, input, out_fea, feat_vgg)
            if self.block_skip:
                records.append(x)

        if self.block_skip:
            out_B = self.c(torch.cat(records, dim=1))
        else:
            out_B = x

        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


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


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


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
        start_time = time.clock()

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

        ESA.total_time += time.clock() - start_time
        return x * m


class SRB(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv = conv_layer(in_channel, in_channel, 3)

    def forward(self, x):
        return x + self.conv(x)


class RepSRB(nn.Module):
    def __init__(self, in_channel, norm_type='spade', num_conv_srb=1, add_ori=True, use_act=True, add_fea=False, vgg_feat=False, **kwargs):
        super().__init__()
        self.add_ori = add_ori
        self.add_fea = add_fea
        self.vgg_feat = vgg_feat
        self.norm_type = norm_type

        conv_in = in_channel + (3 if add_ori else 0) + (in_channel if add_fea else 0)
        self.conv = conv_layer(conv_in, in_channel, 3)

        if norm_type == 'spade':
            if vgg_feat:
                self.norm = SPADE('spadebatch3x3', in_channel, label_nc=128)
            else:
                self.norm = SPADE('spadebatch3x3', in_channel)
        elif norm_type == 'bn':
            self.norm = nn.BatchNorm2d(in_channel)
        elif norm_type == 'in':
            self.norm = nn.InstanceNorm2d(in_channel)
        elif norm_type == 'none':
            self.norm = nn.Identity()

        if use_act:
            self.act = activation('lrelu', neg_slope=0.05)
        else:
            self.act = nn.Identity()

    def forward(self, x, ori_input, out_fea, vgg_feat):
        if self.vgg_feat and self.norm_type == 'spade':
            out = self.act(self.norm(x, vgg_feat))
        else:
            out = self.act(self.norm(x))

        if self.add_ori:
            out = torch.cat([ori_input, out], dim=1)

        if self.add_fea:
            out = torch.cat([out, out_fea], dim=1)

        # return x + self.conv(out)
        return self.conv(out)
        # return x + self.conv(torch.cat([x, ori_input], dim=1))
        # return x + self.conv(x)
        # return self.act(x + self.conv(x))


class Rep_RFDB(nn.Module):
    total_time = 0

    def __init__(self, in_channels, sub_blocks=3, use_esa=True, vgg_feat=False, **kwargs):
        super().__init__()
        self.sub_blocks = sub_blocks
        self.use_esa = use_esa
        self.vgg_feat = vgg_feat

        self.blocks = nn.ModuleList()
        for i in range(self.sub_blocks):
            self.blocks.append(RepSRB(in_channels, vgg_feat=vgg_feat, **kwargs))

        self.act = activation('lrelu', neg_slope=0.05)

        if self.use_esa:
            self.c5 = conv_layer(in_channels * 2, in_channels, 1)
            self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input, ori_input, out_fea, vgg_feat):
        start_time = time.clock()

        x = input
        for i in range(self.sub_blocks):
            x = self.blocks[i](x, ori_input, out_fea, vgg_feat)

        if self.use_esa:
            x = self.c5(torch.cat([input, x], dim=1))
            x = self.esa(x)

        Rep_RFDB.total_time += time.clock() - start_time
        return x


class RFDB(nn.Module):
    total_time = 0

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels

        self.c1_d = conv_layer(in_channels, self.dc, 1)
        self.c1_r = RepSRB(in_channels)
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c2_r = RepSRB(in_channels)
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        self.c3_r = RepSRB(in_channels)
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.esa = ESA(in_channels, nn.Conv2d)

    def forward(self, input, ori_input):
        start_time = time.clock()

        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input, ori_input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1, ori_input)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2, ori_input)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        RFDB.total_time += time.clock() - start_time
        return out_fused


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)
