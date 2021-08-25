import torch
import torch.nn as nn

import model.utils
from model.super_resolution_model import common
from .utils import register_model
from ..layerwise_model import ConvertibleLayer, merge_1x1_and_3x3, ConvLayer, \
    SkipConnectionSubModel, InitializableLayer, ConvertibleModel


@register_model
def EDSR_layerwise(**hparams):
    return EDSR_layerwise_Model(**hparams)


class HeadLayer(ConvertibleLayer):
    def __init__(self, rgb_range, n_colors, n_feats, kernel_size):
        super().__init__()
        self.sub_mean = common.MeanShift(rgb_range)
        self.conv = model.utils.default_conv(n_colors, n_feats, kernel_size)

    def forward(self, x):
        x = self.sub_mean(x[:, 1:])
        return self.conv(x)

    def simplify_layer(self):
        sub_mean = ConvLayer.fromConv2D(self.sub_mean)
        conv = ConvLayer.fromConv2D(self.conv)
        conv = merge_1x1_and_3x3(sub_mean, conv)
        return conv.simplify_layer()


def resBlock(n_feats, kernel_size, act):
    """
    EDSR 的 resBlock 只有中间有个 act，第二层后面是没有act的
    :param n_feats: model width
    :param kernel_size: usually 3
    :param act: act in resBlock
    :return: list of Convertible layers
    """
    conv1 = ConvLayer(n_feats, n_feats, kernel_size, act=act)
    conv2 = ConvLayer(n_feats, n_feats, kernel_size)
    return SkipConnectionSubModel([conv1, conv2], n_feats, skip_connection_bias=1000)


class EDSRTail(InitializableLayer):
    def __init__(self, scale, n_feats, n_colors, kernel_size, rgb_range):
        super().__init__()
        m_tail = [
            common.Upsampler(model.utils.default_conv, scale, n_feats, act=False),
            model.utils.default_conv(n_feats, n_colors, kernel_size)
        ]
        self.tail = nn.Sequential(*m_tail)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        self.n_feats = n_feats
        self.n_colors = n_colors
        self.kernel_size = kernel_size
        self.scale = scale
        self.rgb_range = rgb_range

    def forward(self, x):
        return self.add_mean(self.tail(x[:, 1:]))

    def init_student(self, conv_s, M):
        assert isinstance(conv_s, EDSRTail)
        assert conv_s.scale == self.scale
        assert conv_s.kernel_size == self.kernel_size
        assert conv_s.n_feats == self.n_feats
        assert conv_s.n_colors == self.n_colors
        assert conv_s.rgb_range == self.rgb_range

        import copy

        conv_s.tail = copy.deepcopy(self.tail)
        teacher_conv = ConvLayer.fromConv2D(self.tail[0][0])
        student_conv = copy.deepcopy(teacher_conv)
        teacher_conv.init_student(student_conv, M)
        conv_s.tail[0][0] = student_conv.conv
        return torch.eye(self.n_colors)


class EDSR_layerwise_Model(ConvertibleModel):
    def __init__(self, n_resblocks=16, n_feats=64, nf=None, scale=4, rgb_range=255, n_colors=3, **kwargs):
        super(EDSR_layerwise_Model, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats if nf is None else nf
        kernel_size = 3

        # define head module
        self.sequential_models.append(HeadLayer(rgb_range, n_colors, n_feats, kernel_size))

        # define body module
        for _ in range(n_resblocks):
            self.sequential_models += [resBlock(n_feats, kernel_size, act=nn.ReLU())]
        self.sequential_models.append(ConvLayer(n_feats, n_feats, kernel_size))

        # define tail module
        self.sequential_models.append(EDSRTail(scale, n_feats, n_colors, kernel_size, rgb_range))
