import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basic_cifar_models.utils import register_model


# Credit to https://github.com/akamaster/pytorch_resnet_cifar10


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_1(nn.Module):
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


class BasicBlock_2(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_2, self).__init__()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        planes = x.size(1)//2
        x, shortcut = x[:, :planes], x[:, planes:]

        x = self.bn2(self.conv2(x))
        x += shortcut
        return F.relu(x)


class LastLinearLayer(nn.Module):
    def __init__(self, linear_in, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(linear_in, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class LayerWiseModel(nn.Module):
    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        pass

    def __len__(self):
        pass


class ResNet_CIFAR(LayerWiseModel):
    def __init__(self, num_blocks, num_classes=10, num_filters=(16, 16, 32, 64), option='A'):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = num_filters[0]

        self.sequential_models = nn.ModuleList()

        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.sequential_models.append(nn.Sequential(self.conv1, self.bn1, nn.ReLU()))

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

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        for m in self.sequential_models[start_forward_from: until]:
            x = m(x)
            f_list.append(x)
        return (f_list, x) if with_feature else x

    def __len__(self):
        return len(self.sequential_models)


def _resnet20(expansion=1.0, **kwargs):
    return ResNet_CIFAR(num_blocks=[3, 3, 3], num_filters=[int(i * expansion) for i in [16, 16, 32, 64]], **kwargs)


@register_model
def resnet20_act_wise(**kwargs):
    return _resnet20(expansion=1, **kwargs)
