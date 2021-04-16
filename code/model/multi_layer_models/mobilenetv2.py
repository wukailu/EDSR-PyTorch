"""
MobileNetV2 implementation used in
<Knowledge Distillation via Route Constrained Optimization>
"""

import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv2_T_w', 'mobile_half']

BN = None


class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel, stride, padding, with_feature=True):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.relu = nn.ReLU(inplace=False)
        self.with_feature = with_feature

    def forward(self, x):
        if isinstance(x, tuple):
            pre_acts, acts, x = x
        else:
            pre_acts, acts, x = [], [], x

        x = self.conv(x)
        x = self.bn(x)
        pre_acts.append(x)
        x = self.relu(x)
        acts.append(x)

        if self.with_feature:
            return pre_acts, acts, x
        else:
            return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, with_feature=True):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.with_feature = with_feature

    def forward(self, x):
        if isinstance(x, tuple):
            pre_acts, acts, x = x
        else:
            pre_acts, acts, x = [], [], x

        t = x
        if self.use_res_connect:
            x = t + self.conv(x)
        else:
            x = self.conv(x)
        pre_acts.append(x)
        acts.append(x)

        if self.with_feature:
            return pre_acts, acts, x
        else:
            return x


class MobileNetV2(nn.Module):
    """mobilenetV2"""

    def __init__(self, T,
                 feature_dim,
                 input_size=32,
                 width_mult=1.,
                 remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = Conv_bn_relu(3, input_channel, 3, 2, 1)

        # building inverted residual blocks
        self.blocks = []
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(
                    InvertedResidual(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))
        self.blocks = nn.Sequential(*self.blocks)

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = Conv_bn_relu(input_channel, self.last_channel, 1, 1, 0)

        # building classifier
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(self.last_channel, feature_dim),
        )

        H = input_size // (32 // 2)
        self.avgpool = nn.AvgPool2d(H, ceil_mode=True)

        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        # feat_m.append(self.conv2)
        return feat_m

    def forward(self, x, with_feature=False, pre_act=False):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)

        if isinstance(x, tuple):
            pre_acts, acts, x = x
        else:
            pre_acts, acts, x = [], [], x
        # print(len(pre_acts), print(acts))

        if not self.remove_avg:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f5 = x
        x = self.classifier(x)

        if with_feature:
            if pre_act:
                return pre_acts + [f5], x
            else:
                return acts + [f5], x
        else:
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model


def mobile_half(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)


if __name__ == '__main__':
    test_x = torch.randn(2, 3, 32, 32)

    net = mobile_half(100)

    test_feats, test_logit = net(test_x, with_feature=True, pre_act=False)
    print(len(test_feats))
    for f in test_feats:
        print(f.shape, f.min().item())
    print(test_logit.shape)
