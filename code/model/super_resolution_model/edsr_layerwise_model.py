import torch.nn as nn
from model.super_resolution_model import common
from .utils import register_model, unpack_feature, pack_feature
from .. import LayerWiseModel


@register_model
def EDSR_layerwise(**hparams):
    model = EDSR_layerwise_Model(**hparams)
    return model


class EDSR_layerwise_Model(LayerWiseModel):
    def __init__(self, n_resblocks=16, n_feats=64, nf=None, scale=4, rgb_range=255, n_colors=3, res_scale=1,
                 conv=common.default_conv, **kwargs):
        super(EDSR_layerwise_Model, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats if nf is None else nf
        kernel_size = 3
        act = nn.ReLU(True)

        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        # define head module
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        feat = []
        x = self.sub_mean(x)
        x = self.head(x)
        feat.append(x)

        res = self.body(x)
        feat.append(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        if with_feature:
            return feat, x
        else:
            return x
