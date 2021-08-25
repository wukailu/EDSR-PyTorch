import os
import torch

import model.utils
from model.super_resolution_model import common
import torch.nn as nn
from .utils import register_model, unpack_feature, pack_feature

url = {
    'r16f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr_baseline-a00cab12.pt',
    'r80f64': 'https://cv.snu.ac.kr/research/EDSR/models/mdsr-4a78bedf.pt'
}

@register_model
def MDSR(pretrained=False, **hparams):
    model = MDSR_Model(**hparams)
    if pretrained and model.url is not None:
        dir_model = os.path.join('/data/models')
        os.makedirs(dir_model, exist_ok=True)
        load_from = torch.utils.model_zoo.load_url(model.url, model_dir=dir_model)
        model.load_state_dict(load_from, strict=False)
    return model

class MDSR_Model(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, rgb_range=255, n_colors=3, scale=4, conv=model.utils.default_conv):
        super(MDSR_Model, self).__init__()
        n_resblocks = n_resblocks
        n_feats = n_feats
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale_idx = 0
        self.url = url['r{}f{}'.format(n_resblocks, n_feats)]
        self.sub_mean = common.MeanShift(rgb_range)
        self.add_mean = common.MeanShift(rgb_range, sign=1)

        m_head = [conv(n_colors, n_feats, kernel_size)]

        self.pre_process = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in scale
        ])

        m_body = [
            common.ResBlock(
                conv, n_feats, kernel_size, act=act
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.upsample = nn.ModuleList([
            common.Upsampler(conv, s, n_feats, act=False) for s in scale
        ])

        m_tail = [conv(n_feats, n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        x = self.pre_process[self.scale_idx](x)

        res = self.body(x)
        res += x

        x = self.upsample[self.scale_idx](res)
        x = self.tail(x)
        x = self.add_mean(x)

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

