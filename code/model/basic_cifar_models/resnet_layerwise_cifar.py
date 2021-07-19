import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LayerWiseModel, ConvertibleLayer
from model.basic_cifar_models.utils import register_model

relu_offset = 0  # It's lucky that all feature in resnet is positive


def convbn_to_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    bn.eval()
    out_channel, in_channel, kernel_size, _ = conv.weight.shape

    var = bn.running_var.data
    weight = bn.weight.data
    gamma = weight / (var + bn.eps)

    bias = 0 if conv.bias is None else conv.bias.data

    conv_data = conv.weight.data * gamma.reshape((-1, 1, 1, 1))
    bias = bn.bias.data + (bias - bn.running_mean.data) * gamma

    ret = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True,
                    padding_mode=conv.padding_mode)
    ret.weight.data = conv_data
    ret.bias.data = bias
    return ret


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1(ConvertibleLayer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_1, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = \
                    LambdaLayer(
                        lambda x: F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes // 4, planes // 4], "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        return torch.cat([x, shortcut], dim=1)

    def simplify_layer(self):
        eq_conv = convbn_to_conv(self.conv1, self.bn1)
        channel = eq_conv.out_channels
        conv = nn.Conv2d(eq_conv.in_channels, eq_conv.out_channels * 2, eq_conv.kernel_size, eq_conv.stride,
                         eq_conv.padding, bias=True, padding_mode=eq_conv.padding_mode)
        kernel = torch.zeros_like(conv.weight)
        bias = torch.zeros_like(conv.bias)

        kernel[:channel, ...] = eq_conv.weight.data
        bias[:channel] = eq_conv.bias.data

        if isinstance(self.shortcut, LambdaLayer):
            assert eq_conv.out_channels == eq_conv.in_channels * 2
            for i in range(eq_conv.in_channels):
                kernel[channel + channel // 4 + i, i, eq_conv.kernel_size[0] // 2, eq_conv.kernel_size[1] // 2] = 1
        else:
            # Identity
            assert eq_conv.out_channels == eq_conv.in_channels
            for i in range(eq_conv.in_channels):
                kernel[channel + i, i, eq_conv.kernel_size[0] // 2, eq_conv.kernel_size[1] // 2] = 1

        # make sure the identity can pass relu
        bias[channel:] = relu_offset  # TODO: How will the offset change distillation results?

        conv.weight.data = kernel
        conv.bias.data = bias
        return conv, nn.ReLU()


class BasicBlock_2(ConvertibleLayer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_2, self).__init__()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        planes = x.size(1) // 2
        x, shortcut = x[:, :planes], x[:, planes:]

        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)

    def simplify_layer(self):
        eq_conv = convbn_to_conv(self.conv2, self.bn2)

        out_channel, in_channel, kernel_size, _ = eq_conv.weight.shape
        kernel = torch.zeros((out_channel, in_channel * 2, kernel_size, kernel_size))
        kernel[:, :in_channel, :, :] = eq_conv.weight.data
        for i in range(out_channel):
            kernel[i, i + in_channel, kernel_size // 2, kernel_size // 2] = 1

        conv = nn.Conv2d(in_channel * 2, out_channel, kernel_size=kernel_size, padding=eq_conv.padding)
        conv.weight.data = kernel
        conv.bias.data = eq_conv.bias.data
        return conv, nn.ReLU()


class LastLinearLayer(ConvertibleLayer):
    def __init__(self, linear_in, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(linear_in, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)

    def init_student(self, layer_s, M):
        assert isinstance(layer_s, LastLinearLayer)
        layer_s.linear.weight.data = self.linear.weight.data @ M
        layer_s.linear.bias.data = self.linear.bias
        return torch.diag(torch.ones((self.linear.out_features,)))

    def simplify_layer(self):
        raise NotImplementedError


class ConvBNReLULayer(ConvertibleLayer):
    def __init__(self, in_planes, planes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn1(self.conv1(x)))

    def simplify_layer(self):
        conv = convbn_to_conv(self.conv1, self.bn1)
        return conv, self.relu


class ResNet_CIFAR(LayerWiseModel):
    def __init__(self, num_blocks, num_classes=10, num_filters=(16, 16, 32, 64), option='A'):
        super().__init__()
        self.in_planes = num_filters[0]

        self.sequential_models.append(ConvBNReLULayer(3, num_filters[0]))
        self.sequential_models += self._make_layer(num_filters[1], num_blocks[0], stride=1, option=option)
        self.sequential_models += self._make_layer(num_filters[2], num_blocks[1], stride=2, option=option)
        self.sequential_models += self._make_layer(num_filters[3], num_blocks[2], stride=2, option=option)
        self.sequential_models.append(LastLinearLayer(num_filters[3], num_classes))

    def _make_layer(self, planes, num_blocks, stride, option):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_1(self.in_planes, planes, stride, option))
            layers.append(BasicBlock_2(self.in_planes, planes, stride, option))
            self.in_planes = planes
        return layers


def _resnet20(expansion=1.0, **kwargs):
    return ResNet_CIFAR(num_blocks=[3, 3, 3], num_filters=[int(i * expansion) for i in [16, 16, 32, 64]], **kwargs)


@register_model
def resnet20_layerwise(**kwargs):
    return _resnet20(expansion=1, **kwargs)
