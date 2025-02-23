from typing import Tuple

import torch.nn as nn
import numpy as np
import torch
from model.layerwise_model import ConvertibleLayer


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bn=True):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=not bn))
    if bn:
        result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, version='default',
                 **kwargs):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.version = version
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        assert kernel_size == 3
        assert padding == 1 or padding == (1, 1)

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        elif self.version == 'traditional':  # normal vgg
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
        elif self.version == 'special_ini' or self.version == 'special_ini_fix1':  # normal vgg with special ini
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            fake_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                 padding=padding_11, groups=groups)

            with torch.no_grad():
                self.rbr_dense.conv.weight[:, :, 1, 1] += fake_1x1.weight[:, :, 0, 0]
                if in_channels == out_channels and stride == 1:
                    i = torch.arange(in_channels)
                    self.rbr_dense.conv.weight[i, i, 1, 1] += 1
        elif self.version == 'special_ini_pro':  # normal vgg with special ini
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            fake_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                 padding=padding_11, groups=groups)

            with torch.no_grad():
                self.rbr_dense.conv.weight[:, :, 1, 1] += fake_1x1.weight[:, :, 0, 0]
                i = torch.arange(min(in_channels, out_channels))
                self.rbr_dense.conv.weight[i, i, 1, 1] += 1
        elif self.version == 'special_ini_2xID':  # normal vgg with special ini
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            fake_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                 padding=padding_11, groups=groups)

            with torch.no_grad():
                self.rbr_dense.conv.weight[:, :, 1, 1] += fake_1x1.weight[:, :, 0, 0]
                if in_channels == out_channels and stride == 1:
                    i = torch.arange(in_channels)
                    self.rbr_dense.conv.weight[i, i, 1, 1] += 2
        elif self.version == 'double':  # extra 3x3 and extra 1x1
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
            self.rbr_dense_extra = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                           stride=stride, padding=padding, groups=groups)
            self.rbr_1x1_extra = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                         stride=stride,
                                         padding=padding_11, groups=groups)
        elif self.version == 'moreID' or self.version == 'tripleID' or self.version == 'quadraID' or self.version == 'default':
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
        else:
            raise KeyError("RepVGG version key error!")

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.version == 'special_ini_fix1':
            if self.in_channels == self.out_channels and self.stride == 1:
                with torch.no_grad():
                    i = torch.arange(self.in_channels)
                    self.rbr_dense.conv.weight[i, i, 1, 1] = 1

        outputs = []

        if hasattr(self, 'rbr_1x1'):
            outputs += [self.rbr_1x1(inputs)]
        if hasattr(self, 'rbr_dense'):
            outputs += [self.rbr_dense(inputs)]
        if hasattr(self, 'rbr_dense_extra'):
            outputs += [self.rbr_dense_extra(inputs)]
        if hasattr(self, 'rbr_1x1_extra'):
            outputs += [self.rbr_1x1_extra(inputs)]

        outputs = sum(outputs)
        if hasattr(self, 'rbr_identity'):
            if self.version == 'moreID':
                outputs += self.rbr_identity(inputs) * 2
            elif self.version == 'tripleID':
                outputs += self.rbr_identity(inputs) * 3
            elif self.version == 'quadraID':
                outputs += self.rbr_identity(inputs) * 4
            else:
                outputs += self.rbr_identity(inputs)

        return self.nonlinearity(outputs)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        if hasattr(self.rbr_dense, 'bn'):
            kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        else:
            kernel3x3, bias3x3 = self.rbr_dense.conv.weight, self.rbr_dense.conv.bias

        if hasattr(self.rbr_1x1, 'bn'):
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        else:
            kernel1x1, bias1x1 = self.rbr_1x1.conv.weight, self.rbr_1x1.conv.bias

        if hasattr(self, 'rbr_identity') and isinstance(self.rbr_identity, nn.BatchNorm2d):
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        elif hasattr(self, 'rbr_identity') and isinstance(self.rbr_identity, nn.Identity):
            kernelid, biasid = torch.zeros_like(kernel3x3), 0
            kernelid[:, :, 1, 1] = torch.eye(kernel3x3.size(0))
        else:
            kernelid, biasid = 0, 0
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy()


class LayerwiseRepBlock(ConvertibleLayer, RepVGGBlock):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1,
                 padding_mode='zeros', use_bn=True, **kwargs):
        ConvertibleLayer.__init__(self)
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.nonlinearity = nn.ReLU()

        if out_channels == in_channels and stride == 1:
            if use_bn:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = nn.Identity()
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                 stride=stride, padding=padding, groups=groups, bn=use_bn)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                               padding=padding - kernel_size // 2, groups=groups, bn=use_bn)

    def forward(self, inputs):
        inputs = inputs[:, 1:]
        outputs = self.rbr_1x1(inputs) + self.rbr_dense(inputs)
        if hasattr(self, 'rbr_identity'):
            outputs += self.rbr_identity(inputs)
        return self.nonlinearity(outputs)

    def simplify_layer(self) -> Tuple[nn.Conv2d, nn.Module]:
        kernel, bias = self.get_equivalent_kernel_bias()
        dense = self.rbr_dense.conv
        conv = nn.Conv2d(in_channels=self.in_channels+1, out_channels=self.out_channels, kernel_size=dense.kernel_size,
                         stride=dense.stride, padding=dense.padding, groups=dense.groups, bias=False)
        mid = dense.kernel_size[0] // 2
        conv.weight.data[:, 1:] = kernel
        conv.weight.data[:, 0] = 0
        conv.weight.data[:, 0, mid, mid] = bias
        return conv, self.nonlinearity


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False,
                 version='default', **kwargs):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.version = version
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, version=self.version)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      version=self.version))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False, **kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_A1(deploy=False, **kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_A2(deploy=False, **kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_B0(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_B1(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_B1g2(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy, **kwargs)


def create_RepVGG_B1g4(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy, **kwargs)


def create_RepVGG_B2(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_B2g2(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy, **kwargs)


def create_RepVGG_B2g4(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy, **kwargs)


def create_RepVGG_B3(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy, **kwargs)


def create_RepVGG_B3g2(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy, **kwargs)


def create_RepVGG_B3g4(deploy=False, **kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy, **kwargs)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


#   Use this for converting a customized model with RepVGG as one of its components (e.g., the backbone of a semantic segmentation model)
#   The use case will be like
#   1.  Build train_model. For example, build a PSPNet with a training-time RepVGG as backbone
#   2.  Train train_model or do whatever you want
#   3.  Build deploy_model. In the above example, that will be a PSPNet with an inference-time RepVGG as backbone
#   4.  Call this func
#   ====================== the pseudo code will be like
#   train_backbone = create_RepVGG_B2(deploy=False, **kwargs)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_backbone = create_RepVGG_B2(deploy=True)
#   deploy_pspnet = build_pspnet(backbone=deploy_backbone)
#   whole_model_convert(train_pspnet, deploy_pspnet)
#   segmentation_test(deploy_pspnet)
def whole_model_convert(train_model: torch.nn.Module, deploy_model: torch.nn.Module, save_path=None):
    all_weights = {}
    for name, module in train_model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            all_weights[name + '.rbr_reparam.weight'] = kernel
            all_weights[name + '.rbr_reparam.bias'] = bias
            print('convert RepVGG block')
        else:
            for p_name, p_tensor in module.named_parameters():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    all_weights[full_name] = p_tensor.detach().cpu().numpy()
            for p_name, p_tensor in module.named_buffers():
                full_name = name + '.' + p_name
                if full_name not in all_weights:
                    all_weights[full_name] = p_tensor.cpu().numpy()

    deploy_model.load_state_dict(all_weights)
    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model


#   Use this when converting a RepVGG without customized structures.
#   train_model = create_RepVGG_A0(deploy=False, **kwargs)
#   train train_model
#   deploy_model = repvgg_convert(train_model, create_RepVGG_A0, save_path='repvgg_deploy.pth')
def repvgg_model_convert(model: torch.nn.Module, build_func, save_path=None):
    converted_weights = {}
    for name, module in model.named_modules():
        if hasattr(module, 'repvgg_convert'):
            kernel, bias = module.repvgg_convert()
            converted_weights[name + '.rbr_reparam.weight'] = kernel
            converted_weights[name + '.rbr_reparam.bias'] = bias
        elif isinstance(module, torch.nn.Linear):
            converted_weights[name + '.weight'] = module.weight.detach().cpu().numpy()
            converted_weights[name + '.bias'] = module.bias.detach().cpu().numpy()
    del model

    deploy_model = build_func(deploy=True)
    for name, param in deploy_model.named_parameters():
        print('deploy param: ', name, param.size(), np.mean(converted_weights[name]))
        param.data = torch.from_numpy(converted_weights[name]).float()

    if save_path is not None:
        torch.save(deploy_model.state_dict(), save_path)

    return deploy_model
