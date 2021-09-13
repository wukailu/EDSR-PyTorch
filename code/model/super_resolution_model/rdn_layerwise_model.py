# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import torch
import torch.nn as nn

from . import RDN_Tail
from .utils import register_model
from ..layerwise_model import ConvLayer, ConvertibleLayer, IdLayer, ConcatLayer, \
    SkipConnectionSubModel, ConvertibleModel, SequentialConvertibleSubModel, DenseFeatureFusionSubModel


@register_model
def RDN_layerwise(**hparams):
    return RDN_layerwise_Model(**hparams)


class RDB_Conv_Layerwise(ConvertibleLayer):
    def simplify_layer(self):
        id = IdLayer(self.inChannel, bias=0, act=nn.ReLU())  # TODO: fill this bias
        layer = ConcatLayer(id, self.conv, share_input=True)
        return layer.simplify_layer()

    def __init__(self, inChannels, growRate, kSize=3):
        super().__init__()
        self.inChannel = inChannels
        self.conv = ConvLayer(inChannels, growRate, kSize, stride=1, act=nn.ReLU())

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x[:, 1:], out), 1)


def RDB_Layerwise(growRate0, growRate, nConvLayers, kSize=3):
    convs = []
    for c in range(nConvLayers):
        convs.append(RDB_Conv_Layerwise(growRate0 + c * growRate, growRate, kSize))
    # Local Feature Fusion
    LFF = ConvLayer(growRate0 + nConvLayers * growRate, growRate0, kernel_size=1, stride=1)
    model = SkipConnectionSubModel([*convs, LFF], growRate0, skip_connection_bias=0)  # TODO: fill this bias
    return model


class RDN_layerwise_Model(ConvertibleModel):
    def __init__(self, scale=4, n_feats=64, RDNkSize=3, RDNconfig='B', n_colors=3, **kwargs):
        super().__init__()
        G0 = n_feats
        kSize = RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (6, 4, 32),
            # 'B': (16, 8, 64),
        }[RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = ConvLayer(n_colors, G0, kSize, stride=1)
        self.SFENet2 = ConvLayer(G0, G0, kSize, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB_Layerwise(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = SequentialConvertibleSubModel(
            ConvLayer(self.D * G0, G0, 1, stride=1),
            ConvLayer(G0, G0, kSize, stride=1)
        )

        # Up-sampling net
        self.UPNet = RDN_Tail(n_feats, scale, RDNkSize, n_colors, G, remove_const_channel=True)

        self.append(self.SFENet1)
        backbone = SkipConnectionSubModel([self.SFENet2,
                                           DenseFeatureFusionSubModel(self.RDBs, G0, skip_connection_bias=0),
                                           self.GFF], G0, skip_connection_bias=0)
        self.append(backbone)
        self.append(self.UPNet)
