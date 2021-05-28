import os
import torch
import torch.nn as nn
from model.super_resolution_model import common
from .utils import register_model, unpack_feature, pack_feature


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}

@register_model
def EDSR(pretrained=False, **hparams):
    model = EDSR_Model(**hparams)
    if pretrained and model.url is not None:
        dir_model = os.path.join('/data/models')
        os.makedirs(dir_model, exist_ok=True)
        load_from = torch.utils.model_zoo.load_url(model.url, model_dir=dir_model)
        model.load_state_dict(load_from, strict=False)
    return model

class EDSR_Model(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, scale=4, rgb_range=255, n_colors=3, res_scale=1, conv=common.default_conv, **kwargs):
        super(EDSR_Model, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
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

    def forward(self, x, with_feature=False):
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
            return x, feat
        else:
            return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

