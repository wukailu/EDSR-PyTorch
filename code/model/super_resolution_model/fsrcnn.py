import math
from torch import nn
from .utils import register_model


@register_model
def FSRCNN(**kwargs):
    model = FSRCNN_Model(**kwargs)
    return model


class FSRCNN_Model(nn.Module):
    def __init__(self, scale, n_colors=3, n_feat=56, s=12, m=4, **kwargs):
        super(FSRCNN_Model, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(n_colors, n_feat, kernel_size=5, padding=5 // 2),
            nn.PReLU(n_feat)
        )
        self.mid_part = [nn.Conv2d(n_feat, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, n_feat, kernel_size=1), nn.PReLU(n_feat)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(n_feat, n_colors, kernel_size=9, stride=scale, padding=9 // 2,
                                            output_padding=scale - 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
