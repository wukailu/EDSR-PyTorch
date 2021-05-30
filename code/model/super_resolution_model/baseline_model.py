import torch
import torch.nn as nn
from .utils import register_model, unpack_feature, pack_feature


@register_model
def DirectScale(**kwargs):
    model = DirectScale_Model(**kwargs)
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
