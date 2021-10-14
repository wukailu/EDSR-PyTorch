import torch
from torch import nn
from .utils import register_model


@register_model
def SRCNN(**kwargs):
    model = SRCNN_Model(**kwargs)
    return model


class SRCNN_Model(nn.Module):
    def __init__(self, n_colors=1, scale=4, **kwargs):
        super(SRCNN_Model, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(n_colors, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, n_colors, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        with torch.no_grad():
            x = torch.nn.functional.interpolate(x, scale_factor=self.scale, mode='bicubic')
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
