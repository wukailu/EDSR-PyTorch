import torch.nn as nn
import torch
from model.super_resolution_model import common
from .utils import register_model
from .. import LayerWiseModel, ConvertibleLayer, merge_1x1_and_3x3


@register_model
def EDSR_layerwise(**hparams):
    model = EDSR_layerwise_Model(**hparams)
    return model


# TODO: remove this class by mean-shift
# How to use mean-shift correctly when convs are used on edges.
class PartialAct(nn.Module):
    def __init__(self, act_range, act=nn.ReLU()):
        super().__init__()
        self.act_range = act_range
        self.act = act

    def forward(self, x):
        if self.act_range[0] is None:
            assert self.act_range[1] is not None
            x0 = x[:, :self.act_range[1]]
            x1 = x[:, self.act_range[1]:]
            x0 = self.act(x0)
            return torch.cat([x0, x1], dim=1)
        elif self.act_range[1] is None:
            assert self.act_range[0] is not None
            x0 = x[:, :self.act_range[0]]
            x1 = x[:, self.act_range[0]:]
            x1 = self.act(x1)
            return torch.cat([x0, x1], dim=1)
        else:
            x0 = x[:, :self.act_range[0]]
            x1 = x[:, self.act_range[0]:self.act_range[1]]
            x2 = x[self.act_range[1]:]
            x1 = self.act(x1)
            return torch.cat([x0, x1, x2], dim=1)


class HeadLayer(ConvertibleLayer):
    def __init__(self, rgb_range, n_colors, n_feats, kernel_size):
        super().__init__()
        self.sub_mean = common.MeanShift(rgb_range)
        self.conv = common.default_conv(n_colors, n_feats, kernel_size)

    # TODO: special case for bias-free conv
    def forward(self, x):
        x = self.sub_mean(x)
        return self.conv(x)

    def simplify_layer(self):
        return merge_1x1_and_3x3(self.sub_mean, self.conv), nn.Identity()


class ConvLayer(ConvertibleLayer):
    def __init__(self, in_channel, out_channel, kernel_size, act: nn.Module = nn.Identity(), bias=True):
        super().__init__()
        self.conv = common.default_conv(in_channel, out_channel, kernel_size, bias=bias)
        self.act = act

    # TODO: special case for bias-free conv
    def simplify_layer(self):
        return self.conv, self.act

    def forward(self, x):
        return self.act(self.conv(x))


class IdLayer(ConvertibleLayer):
    def __init__(self, channel):
        super().__init__()
        self.conv = common.default_conv(channel, channel, 1, bias=False)
        self.conv.weight.data = torch.eye(channel).view((channel, channel, 1, 1))
        self.conv.weight.requires_grad = False

    def simplify_layer(self):
        return self.conv, nn.Identity()

    def forward(self, x):
        return x


class ConcatLayer(ConvertibleLayer):
    def __init__(self, layer1, layer2, share_input=False, sum_output=False, act: nn.Module = nn.Identity()):
        super().__init__()
        assert isinstance(layer1, ConvertibleLayer)
        assert isinstance(layer2, ConvertibleLayer)
        self.eq_conv1: nn.Conv2d = layer1.simplify_layer()[0]
        self.eq_conv2: nn.Conv2d = layer2.simplify_layer()[0]
        self.layer1 = layer1
        self.layer2 = layer2
        self.share_input = share_input
        self.sum_output = sum_output
        self.act = act

    def forward(self, x):
        if self.share_input:
            x1, x2 = x, x
        else:
            assert x.size(1) == self.eq_conv1.in_channels + self.eq_conv2.in_channels
            x1, x2 = x[:, :self.eq_conv1.in_channels], x[:, self.eq_conv1.in_channels:]

        x1, x2 = self.layer1(x1), self.layer2(x2)

        if self.sum_output:
            assert x1.shape == x2.shape
            ret = x1 + x2
        else:
            ret = torch.cat([x1, x2], dim=1)
        return self.act(ret)

    # TODO: special case for bias-free conv
    def simplify_layer(self):
        assert self.eq_conv1.kernel_size[0] == self.eq_conv1.kernel_size[1]
        assert self.eq_conv2.kernel_size[0] == self.eq_conv2.kernel_size[1]
        assert self.eq_conv1.padding[0] == self.eq_conv1.padding[1]
        assert self.eq_conv2.padding[0] == self.eq_conv2.padding[1]
        assert self.eq_conv1.kernel_size[0] % 2 == 1
        assert self.eq_conv2.kernel_size[0] % 2 == 1
        assert self.eq_conv1.kernel_size[0] // 2 == self.eq_conv1.padding[0]
        assert self.eq_conv2.kernel_size[0] // 2 == self.eq_conv2.padding[0]
        assert self.eq_conv1.stride == self.eq_conv2.stride
        assert self.eq_conv1.padding_mode == self.eq_conv2.padding_mode

        if self.share_input:
            assert self.eq_conv1.in_channels == self.eq_conv2.in_channels
            in_channels = self.eq_conv1.in_channels
        else:
            in_channels = self.eq_conv1.in_channels + self.eq_conv2.in_channels

        if self.sum_output:
            assert self.eq_conv1.out_channels == self.eq_conv2.out_channels
            out_channels = self.eq_conv1.out_channels
        else:
            out_channels = self.eq_conv1.out_channels + self.eq_conv2.out_channels

        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=max(self.eq_conv1.kernel_size, self.eq_conv2.kernel_size),
                         padding=max(self.eq_conv1.padding, self.eq_conv2.padding),
                         padding_mode=self.eq_conv1.padding_mode,
                         bias=True)

        bias1 = self.eq_conv1.bias.data if self.eq_conv1.bias is not None else torch.zeros(
            (self.eq_conv1.out_channels,))
        bias2 = self.eq_conv2.bias.data if self.eq_conv2.bias is not None else torch.zeros(
            (self.eq_conv2.out_channels,))
        if self.sum_output:
            conv.bias.data = bias1 + bias2
        else:
            conv.bias.data = torch.cat([bias1, bias2], dim=0)

        kernel = torch.zeros_like(conv.weight)
        slice1 = slice((conv.kernel_size[0] - self.eq_conv1.kernel_size[0]) // 2,
                       (conv.kernel_size[0] + self.eq_conv1.kernel_size[0]) // 2)
        slice2 = slice((conv.kernel_size[0] - self.eq_conv2.kernel_size[0]) // 2,
                       (conv.kernel_size[0] + self.eq_conv2.kernel_size[0]) // 2)

        slice_in = slice(None, self.eq_conv1.in_channels) if self.share_input else slice(self.eq_conv1.in_channels,
                                                                                         None)
        slice_out = slice(None, self.eq_conv1.out_channels) if self.sum_output else slice(self.eq_conv1.out_channels,
                                                                                          None)
        kernel[:self.eq_conv1.out_channels, :self.eq_conv1.in_channels, slice1, slice1] += self.eq_conv1.weight.data
        kernel[slice_out, slice_in, slice2, slice2] += self.eq_conv2.weight.data
        conv.weight.data = kernel

        return conv, self.act


def resBlock(n_feats, kernel_size, act):
    conv1 = ConvLayer(n_feats, n_feats, kernel_size, act=act)
    conv2 = ConvLayer(n_feats, n_feats, kernel_size)
    id1 = IdLayer(n_feats)
    id2 = IdLayer(n_feats)
    return ConcatLayer(conv1, id1, share_input=True, act=PartialAct(act_range=(None, n_feats))), \
           ConcatLayer(conv2, id2, sum_output=True, act=nn.ReLU())


class EDSRTail(ConvertibleLayer):
    def __init__(self, scale, n_feats, n_colors, kernel_size, rgb_range):
        super().__init__()
        m_tail = [
            common.Upsampler(common.default_conv, scale, n_feats, act=False),
            common.default_conv(n_feats, n_colors, kernel_size)
        ]
        self.tail = nn.Sequential(*m_tail)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        self.n_feats = n_feats
        self.n_colors = n_colors
        self.kernel_size = kernel_size
        self.scale = scale
        self.rgb_range = rgb_range

    def forward(self, x):
        return self.add_mean(self.tail(x))

    def init_student(self, conv_s, M):
        assert isinstance(conv_s, EDSRTail)
        assert conv_s.scale == self.scale
        assert conv_s.kernel_size == self.kernel_size
        assert conv_s.n_feats == self.n_feats
        assert conv_s.n_colors == self.n_colors
        assert conv_s.rgb_range == self.rgb_range

        import copy
        from model import matmul_on_first_two_dim

        conv_s.tail = copy.deepcopy(self.tail)
        # print(f"conv_s.tail[0][0].weight.data shape, {conv_s.tail[0][0].weight.data.shape}")
        # print(f"M shape, {M.shape}")
        conv_s.tail[0][0].weight.data = matmul_on_first_two_dim(conv_s.tail[0][0].weight.data, M)
        # TODO: fix bias
        return torch.eye(self.n_colors)

    def simplify_layer(self):
        raise NotImplementedError()


class EDSR_layerwise_Model(LayerWiseModel):
    def __init__(self, n_resblocks=16, n_feats=64, nf=None, scale=4, rgb_range=255, n_colors=3, **kwargs):
        super(EDSR_layerwise_Model, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats if nf is None else nf
        kernel_size = 3

        # define head module
        self.sequential_models.append(HeadLayer(rgb_range, n_colors, n_feats, kernel_size))

        # define body module
        for _ in range(n_resblocks):
            self.sequential_models += list(resBlock(n_feats, kernel_size, act=nn.ReLU(True)))
        self.sequential_models.append(ConvLayer(n_feats, n_feats, kernel_size))

        # define tail module
        self.sequential_models.append(EDSRTail(scale, n_feats, n_colors, kernel_size, rgb_range))
