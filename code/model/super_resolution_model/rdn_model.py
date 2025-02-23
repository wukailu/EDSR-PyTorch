# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from .utils import register_model

import torch
import torch.nn as nn

from ..layerwise_model import InitializableLayer, ConvLayer


@register_model
def RDN(**hparams):
    return RDN_Model(**hparams)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN_Tail(InitializableLayer):
    def __init__(self, n_feats, scale, RDNkSize=3, n_colors=3, G=64, remove_const_channel=True):
        super().__init__()
        self.remove_const_channel = remove_const_channel
        self.n_colors = n_colors
        if scale == 2 or scale == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, G * scale * scale, RDNkSize, padding=(RDNkSize - 1) // 2, stride=1),
                nn.PixelShuffle(scale),
                nn.Conv2d(G, n_colors, RDNkSize, padding=(RDNkSize - 1) // 2, stride=1)
            ])
        elif scale == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(n_feats, G * 4, RDNkSize, padding=(RDNkSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, RDNkSize, padding=(RDNkSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, n_colors, RDNkSize, padding=(RDNkSize - 1) // 2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        if self.remove_const_channel:
            x = x[:, 1:]
        return self.UPNet(x)

    def init_student(self, conv_s, M):
        assert isinstance(conv_s, RDN_Tail)
        import copy

        conv_s.UPNet = copy.deepcopy(self.UPNet)
        teacher_conv = ConvLayer.fromConv2D(self.UPNet[0])
        student_conv = copy.deepcopy(teacher_conv)
        teacher_conv.init_student(student_conv, M)
        conv_s.UPNet[0] = student_conv.conv
        conv_s.remove_const_channel = False
        return torch.eye(self.n_colors)


class RDN_Model(nn.Module):
    def __init__(self, scale=4, n_feats=64, RDNkSize=3, RDNconfig='B', n_colors=3, **kwargs):
        super(RDN_Model, self).__init__()
        G0 = n_feats
        kSize = RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = RDN_Tail(n_feats, scale, RDNkSize, n_colors, G, remove_const_channel=False)

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x += f__1

        return self.UPNet(x)
