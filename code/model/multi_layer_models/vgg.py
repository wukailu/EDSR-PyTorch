"""VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei
"""
import torch.nn as nn
import math

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg8', 'vgg8_bn'
]


class Sub_block(nn.Module):
    def __init__(self, in_channel, out_channel, batch_norm=False, with_feature=True):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=False)
        self.with_feature = with_feature

    def forward(self, x):
        if isinstance(x, tuple):
            pre_acts, acts, x = x
        else:
            pre_acts, acts, x = [], [], x

        x = self.conv2d(x)
        if self.bn:
            x = self.bn(x)
        pre_acts.append(x)
        x = self.relu(x)
        acts.append(x)

        if self.with_feature:
            return pre_acts, acts, x
        else:
            return x


class Sp_Pool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.with_feature = True

    def forward(self, x):
        if isinstance(x, tuple):
            pre_acts, acts, x = x
        else:
            pre_acts, acts, x = [], [], x

        x = self.pool(x)

        if self.with_feature:
            return pre_acts, acts, x
        else:
            return x


class VGG(nn.Module):

    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = Sp_Pool()
        self.pool1 = Sp_Pool()
        self.pool2 = Sp_Pool()
        self.pool3 = Sp_Pool()
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.with_feature = False

    def get_feat_modules(self):
        feat_m: nn.ModuleList = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def forward(self, x, with_feature=None, pre_act=False):
        if with_feature is None:
            with_feature = self.with_feature

        h = x.shape[2]
        x = self.block0(x)
        x = self.pool0(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)

        if isinstance(x, tuple):
            pre_acts, acts, x = x
        else:
            pre_acts, acts, x = [], [], x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        logits = x
        x = self.classifier(x)

        if with_feature:
            if pre_act:
                return pre_acts + [logits], x
            else:
                return acts + [logits], x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [Sp_Pool()]
            else:
                layers += [Sub_block(in_channels, v, batch_norm)]
                in_channels = v
        return nn.Sequential(*layers)

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


config = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    """
    model = VGG(config['S'], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    """
    model = VGG(config['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    """
    model = VGG(config['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(config['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    """
    model = VGG(config['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(config['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    """
    model = VGG(config['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(config['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    """
    model = VGG(config['E'], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(config['E'], batch_norm=True, **kwargs)
    return model


if __name__ == '__main__':
    import torch

    test_x = torch.randn(2, 3, 32, 32).requires_grad_(True)
    net = vgg8_bn(num_classes=100)
    from model.model_utils import freeze
    freeze(net)
    feats, test_logit = net(test_x, with_feature=True, pre_act=False)

    print(len(feats))
    for f in feats:
        print(f.shape, f.min().item())
    print(test_logit.shape)

    feature_grads = {}

    def get_hook(key):
        def hook(grad):
            feature_grads[key] = grad

        return hook

    for idx, f in enumerate(feats):
        f.register_hook(get_hook(idx))

    test_logit.sum().backward()

    for i in range(len(feats)):
        print(feature_grads[i].shape, feature_grads[i].min().item())
