from .resnet import *
from .resnetv2 import *
from .wrn import *
from .vgg import *
from .mobilenetv2 import *
from .repvgg import RepVGG_A1, RepVGG_A1_old

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnet50': ResNet50,
    'resnet18': ResNet18,
    'wrn_16_1': wrn16x1,
    'wrn_16_2': wrn16x2,
    'wrn_40_1': wrn40x1,
    'wrn_40_2': wrn40x2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'mobilenetv2': mobile_half,
    'repvgg_a1': RepVGG_A1,
    'vgg_a1': RepVGG_A1_old
}
