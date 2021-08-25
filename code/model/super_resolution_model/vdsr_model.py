import model.utils
from model.super_resolution_model import common
from .utils import register_model, unpack_feature, pack_feature
import torch.nn as nn


@register_model
def VDSR(**hparams):
    return VDSR_Model(**hparams)


class VDSR_Model(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, rgb_range=255, n_colors=3, conv=model.utils.default_conv, **kwargs):
        super(VDSR_Model, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        def basic_block(in_channels, out_channels, act):
            return common.BasicBlock(
                conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_body = []
        m_body.append(basic_block(n_colors, n_feats, nn.ReLU(True)))
        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(n_feats, n_feats, nn.ReLU(True)))
        m_body.append(basic_block(n_feats, n_colors, None))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.sub_mean(x)
        res = self.body(x)
        res += x
        x = self.add_mean(res)

        return x
