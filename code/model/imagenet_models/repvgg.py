import torch.nn as nn
import numpy as np
import torch
from model.imagenet_models.utils import register_model


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


def bn_relu_conv(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('bn', nn.BatchNorm2d(num_features=in_channels))
    result.add_module('relu', nn.ReLU(inplace=True))
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    return result


def conv_bn_relu(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    result.add_module('relu', nn.ReLU(inplace=True))
    return result


class Multiplication(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, input):
        return input * self.weight


class MultiplicationPro(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((1, channels, 1, 1)), requires_grad=True)

    def forward(self, input):
        return input * self.weight


class MulConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, **kwargs):
        super().__init__()
        self.has_id = in_channels == out_channels and (stride == 1 or stride == (1, 1))
        self.conv = conv_multpro(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, **kwargs)

    def forward(self, input, *inputs, **kwargs):
        if self.has_id:
            return input + self.conv(input)
        else:
            return self.conv(input)


def conv_mult(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('mult', Multiplication())
    return result


def conv_multpro(in_channels, out_channels, kernel_size, stride, padding=0, groups=1, **kwargs):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=False))
    result.add_module('mult', MultiplicationPro(out_channels))
    return result


class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, version='default',
                 record=False, track_sum=False, scale_up=1, **kwargs):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.track_sum = track_sum
        self.record = record
        self.scale_up = scale_up
        self.last_state = {}
        self.record_data = {}  # , '3x3_grad_diff_cos': []

        self.version = version
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode

        if record:
            print("recording output difference.")

        assert kernel_size == 3
        assert padding == 1

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
        elif self.version == 'bn_relu_conv':  # presnet vgg
            self.alpha_schedule = 1
            if out_channels == in_channels and stride == 1:
                self.real_identity = nn.Identity()
            self.rbr_dense = bn_relu_conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups)
        elif self.version == 'conv_mult_bn':  # new structure
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.Identity()
            self.rbr_dense = conv_mult(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, groups=groups)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.version == 'conv_multpro_bn':  # new structure
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.Identity()
            self.rbr_dense = conv_multpro(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.version == 'conv_multpro_all':  # new structure
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = MultiplicationPro(out_channels)
            self.rbr_dense = conv_multpro(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.version == 'new_structure':  # new structure
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.Identity()
            self.rbr_dense = conv_multpro(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_multpro(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=stride, padding=padding_11, groups=groups)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.version == 'like_repvgg':  # new structure
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = MultiplicationPro(out_channels)
            self.rbr_dense = conv_multpro(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_multpro(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                        stride=stride, padding=padding_11, groups=groups)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.version == 'conv_mult_mult_bn':  # new structure
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = Multiplication()
            self.rbr_dense = conv_mult(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding, groups=groups)
            self.bn = nn.BatchNorm2d(num_features=out_channels)
        elif self.version in ('conv_bn_relu', 'conv_bn_relu_prob'):  # resnet vgg
            self.alpha_schedule = 1
            if out_channels == in_channels and stride == 1:
                self.real_identity = nn.Identity()
            self.rbr_dense = conv_bn_relu(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, groups=groups)
        elif self.version == 'repVGG_pro':
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
                self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                       padding=padding_11, groups=groups)
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
        elif self.version == 'repVGG_3x3_ID':
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
        elif self.version == 'sum_bn_relu':  # bn is after sum
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.Identity()
            self.bn = nn.BatchNorm2d(num_features=out_channels)
            self.rbr_dense = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                       bias=False)
            self.rbr_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=1, stride=stride, padding=padding_11, groups=groups,
                                     bias=False)
        elif self.version in (
                'special_ini', 'special_ini_fix1', 'special_ini_fix1_flexible', 'special_ini_2xID',
                'special_ini_smallID'):
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            fake_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                 padding=padding_11, groups=groups)

            if self.version == 'special_ini_fix1_flexible':
                self.alpha_schedule = 1

            if self.version == 'special_ini_2xID':
                adds = 2
            elif self.version == 'special_ini_smallID':
                adds = 0.2
            else:
                adds = 1
            with torch.no_grad():
                self.rbr_dense.conv.weight[:, :, 1, 1] += fake_1x1.weight[:, :, 0, 0]
                if in_channels == out_channels and stride == 1:
                    i = torch.arange(in_channels)
                    self.rbr_dense.conv.weight[i, i, 1, 1] += adds
        elif self.version == 'special_ini_pro':  # normal vgg with special ini
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            fake_1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                 padding=padding_11, groups=groups)

            with torch.no_grad():
                self.rbr_dense.conv.weight[:, :, 1, 1] += fake_1x1.weight[:, :, 0, 0]
                i = torch.arange(min(in_channels, out_channels))
                self.rbr_dense.conv.weight[i, i, 1, 1] += 1
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

    def normal_forward(self, inputs, hook=None):
        if self.version == 'special_ini_fix1' or (
                self.version == 'special_ini_fix1_flexible' and self.alpha_schedule > 0.05):
            if self.in_channels == self.out_channels and self.stride == 1:
                with torch.no_grad():
                    i = torch.arange(self.in_channels)
                    self.rbr_dense.conv.weight[i, i, 1, 1] = 1

        outputs = []
        output_names = []

        if hasattr(self, 'rbr_1x1'):
            outputs += [self.rbr_1x1(inputs)]
            output_names += ['rbr_1x1']
        if hasattr(self, 'rbr_dense'):
            outputs += [self.rbr_dense(inputs)]
            output_names += ['rbr_dense']
        if hasattr(self, 'rbr_dense_extra'):
            outputs += [self.rbr_dense_extra(inputs)]
            output_names += ['rbr_dense_extra']
        if hasattr(self, 'rbr_1x1_extra'):
            outputs += [self.rbr_1x1_extra(inputs)]
            output_names += ['rbr_1x1_extra']
        if hasattr(self, 'rbr_identity'):
            if self.version == 'moreID':
                outputs += [self.rbr_identity(inputs) * 2]
            elif self.version == 'tripleID':
                outputs += [self.rbr_identity(inputs) * 3]
            elif self.version == 'quadraID':
                outputs += [self.rbr_identity(inputs) * 4]
            else:
                outputs += [self.rbr_identity(inputs)]
            output_names += ['rbr_identity']

        if self.track_sum:
            for i, x in enumerate(outputs):
                with torch.no_grad():
                    norm = torch.mean(torch.norm(torch.flatten(x, start_dim=1), p=None, dim=1).data.cpu())
                name = output_names[i]
                if name not in self.record_data:
                    self.record_data[name] = []
                self.record_data[name].append(norm)

        if hook is not None:
            hook(outputs, output_names)

        outputs = sum(outputs)

        if hasattr(self, 'bn'):
            outputs = self.bn(outputs)

        if self.version in ('bn_relu_conv', 'conv_bn_relu', 'conv_bn_relu_prob'):
            if hasattr(self, 'real_identity'):
                if self.version != 'conv_bn_relu_prob' or torch.rand(1) < self.alpha_schedule:
                    outputs += self.real_identity(inputs) * self.alpha_schedule
        else:
            outputs = self.nonlinearity(outputs)

        if self.scale_up != 1:
            outputs = outputs * self.scale_up

        return outputs

    def forward(self, inputs, with_hook=False, hook=None):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.record and torch.is_grad_enabled():
            if len(self.last_state) == 0:
                self.last_state = {k: v.data.clone() for k, v in self.state_dict().items()}
                outputs = self.normal_forward(inputs, hook=hook)
            else:
                current_state = {k: v.data.clone() for k, v in self.state_dict().items()}
                self.load_state_dict(self.last_state)
                with torch.no_grad():
                    outputs_ref = self.normal_forward(inputs)
                self.load_state_dict(current_state)
                self.last_state = current_state

                outputs = self.normal_forward(inputs, hook=hook)

                with torch.no_grad():
                    # sim = torch.norm(torch.flatten(outputs, start_dim=1) - torch.flatten(outputs_ref, start_dim=1), p=None, dim=1)
                    sim = torch.cosine_similarity(torch.flatten(outputs, start_dim=1),
                                                  torch.flatten(outputs_ref, start_dim=1))
                    if 'forward_diff' not in self.record_data:
                        self.record_data['forward_diff'] = []
                    self.record_data['forward_diff'].append(torch.mean(sim).data.cpu())
        else:
            outputs = self.normal_forward(inputs, hook=hook)

        return outputs

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel, bias = self._fuse_bn_tensor(self.rbr_dense)
        if hasattr(self, 'rbr_1x1'):
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
            kernel += self._pad_1x1_to_3x3_tensor(kernel1x1)
            bias += bias1x1
        if hasattr(self, 'rbr_identity'):
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
            kernel += kernelid
            bias += biasid
        return kernel, bias

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
        return kernel.detach().cpu(), bias.detach().cpu()

    def do_repvgg_convert(self):
        kernel, bias = self.repvgg_convert()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
                                     kernel_size=self.kernel_size,
                                     stride=self.stride,
                                     padding=self.padding, dilation=self.dilation, groups=self.groups, bias=True,
                                     padding_mode=self.padding_mode)
        self.rbr_reparam.weight = torch.nn.Parameter(kernel, requires_grad=True)
        self.rbr_reparam.bias = torch.nn.Parameter(bias, requires_grad=True)
        for attr in ['rbr_1x1', 'rbr_dense', 'rbr_identity']:
            if hasattr(self, attr):
                delattr(self, attr)

    def to_traditional(self):
        device = self.rbr_dense.conv.weight.device
        kernel, bias = self.repvgg_convert()
        self.rbr_dense.conv.weight = torch.nn.Parameter(kernel, requires_grad=True)
        self.rbr_dense.conv.bias = torch.nn.Parameter(bias, requires_grad=True)
        self.rbr_dense.bn = nn.BatchNorm2d(num_features=self.out_channels)
        for attr in ['rbr_1x1', 'rbr_identity']:
            if hasattr(self, attr):
                delattr(self, attr)
        self.to(device)


class RepVGG(nn.Module):

    def __init__(self, num_blocks, num_classes=1000, width_multiplier=None, override_groups_map=None, deploy=False,
                 traditional=False, version='default', pretrained=None, **kwargs):
        super(RepVGG, self).__init__()

        assert len(width_multiplier) == 4

        self.deploy = deploy
        self.kwargs = kwargs
        self.version = version
        self.traditional = traditional
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=self.deploy, version=version, **kwargs)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

        if isinstance(pretrained, str):
            state = torch.load(pretrained)
            self.load_state_dict(state)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy,
                                      version=self.version, **self.kwargs))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x, with_feature=False, pre_act=False):
        out = self.stage0(x)
        f0 = out
        out = self.stage1(out)
        f1 = out
        out = self.stage2(out)
        f2 = out
        out = self.stage3(out)
        f3 = out
        out = self.stage4(out)
        f4 = out
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.linear(out)

        if with_feature:
            return [f0, f1, f2, f3, f4, f5], out
        return out


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}


@register_model
def RepVGG_A0(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None,
                  **kwargs)


@register_model
def RepVGG_A1(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, **kwargs)


@register_model
def RepVGG_A1_double(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  version='double', **kwargs)


@register_model
def RepVGG_A1_more_ID(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  version='moreID', **kwargs)


@register_model
def RepVGG_A1_triple_ID(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  version='tripleID', **kwargs)


@register_model
def RepVGG_A1_quadra_ID(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  version='quadraID', **kwargs)


@register_model
def RepVGG_A1_old(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None,
                  version='traditional', **kwargs)


def RepVGG_A2(**kwargs):
    return RepVGG(num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None,
                  **kwargs)


@register_model
def RepVGG_B0(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, **kwargs)


def RepVGG_B1(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=None, **kwargs)


def RepVGG_B1g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, **kwargs)


def RepVGG_B1g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, **kwargs)


def RepVGG_B2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None,
                  **kwargs)


def RepVGG_B2g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map,
                  **kwargs)


def RepVGG_B2g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map,
                  **kwargs)


def RepVGG_B3(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=None, **kwargs)


def RepVGG_B3g2(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, **kwargs)


def RepVGG_B3g4(**kwargs):
    return RepVGG(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, **kwargs)


func_dict = {
    'RepVGG-A0': RepVGG_A0,
    'RepVGG-A1': RepVGG_A1_old,
    'RepVGG-A2': RepVGG_A2,
    'RepVGG-B0': RepVGG_B0,
    'RepVGG-B1': RepVGG_B1,
    'RepVGG-B1g2': RepVGG_B1g2,
    'RepVGG-B1g4': RepVGG_B1g4,
    'RepVGG-B2': RepVGG_B2,
    'RepVGG-B2g2': RepVGG_B2g2,
    'RepVGG-B2g4': RepVGG_B2g4,
    'RepVGG-B3': RepVGG_B3,
    'RepVGG-B3g2': RepVGG_B3g2,
    'RepVGG-B3g4': RepVGG_B3g4,
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
#   train_backbone = RepVGG_B2(deploy=False)
#   train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
#   train_pspnet = build_pspnet(backbone=train_backbone)
#   segmentation_train(train_pspnet)
#   deploy_backbone = RepVGG_B2(deploy=True)
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
#   train_model = RepVGG_A0(deploy=False)
#   train train_model
#   deploy_model = repvgg_convert(train_model, RepVGG_A0, save_path='repvgg_deploy.pth')
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
