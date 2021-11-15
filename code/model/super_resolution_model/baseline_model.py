import torch
import torch.nn as nn
from .utils import register_model
import torch.nn.functional as F


@register_model
def DirectScale(**kwargs):
    model = DirectScale_Model(**kwargs)
    return model


@register_model
def Bicubic(**kwargs):
    model = Bicubic_Model(**kwargs)
    return model


class DirectScale_Model(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, scale=4, **kwargs):
        super().__init__()
        assert in_nc == out_nc
        self.out_nc = out_nc
        self.scale = scale
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, input: torch.Tensor):
        inp = torch.cat([torch.stack([input[:, c, :, :]] * (self.scale ** 2), dim=1) for c in range(self.out_nc)],
                        dim=1)
        out = self.shuffle(inp)
        return out


class Bicubic_Model(nn.Module):
    def __init__(self, scale=4, **kwargs):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bicubic', align_corners=False)
