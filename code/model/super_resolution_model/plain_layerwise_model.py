import torch
import torch.nn as nn

from . import common
from .utils import register_model
from .. import SR_conv_init
from ..layerwise_model import LayerWiseModel


@register_model
def Plain_layerwise(in_nc=3, n_feats=50, nf=None, widths=None, num_modules=16, out_nc=3, scale=4, tail='easy', mean_shift=False,
                    rgb_range=255, n_colors=3, **kwargs):
    nf = n_feats if nf is None else nf
    input_transform = common.MeanShift(rgb_range, sign=-1) if mean_shift else None
    output_transform = common.MeanShift(rgb_range, sign=1) if mean_shift else None

    if widths is None:
        widths = [in_nc] + [nf] * num_modules

    if tail == 'easy':
        tailModule = EasyScale(scale, output_transform=output_transform)
        widths += [(scale ** 2) * out_nc]
    elif tail == 'edsr':
        from model.super_resolution_model.edsr_layerwise_model import EDSRTail
        tailModule = EDSRTail(scale, n_feats, n_colors, kernel_size=3, rgb_range=rgb_range)
        widths += [n_feats]
    else:
        raise NotImplementedError()

    model = Plain_layerwise_Model(widths=widths, input_transform=input_transform, **kwargs)
    model.append_tail(tailModule)

    return model


# TODO: implement this as ConvertibleModel
class Plain_layerwise_Model(LayerWiseModel):
    def __init__(self, widths, layerType='normal_no_bn', input_transform=None, f_lists=None, add_ori=False,
                 stack_output=False, square_ratio=0, square_num=0, square_layer_strategy=0, square_before_relu=False,
                 add_ori_interval=1, **kwargs):
        """
        :arg widths width of each feature map, start from data, end at the one before tail. e.x. [3, 64, 64, 128, 200]
        :arg add_ori if this is true, there will be 3 more channel on input, which is original input data
        :arg add_ori_interval this one specify the interval of add_ori, default to be 1
        :arg stack_output if this is true, all feature maps will be stacked and pass by a 1x1 conv to generate output before tail,
        otherwise output is just the result from last conv
        :type stack_output: bool
        :type add_ori: bool
        :type add_ori_interval: int
        """
        super().__init__()
        self.layerType = layerType
        if add_ori:
            self.add_ori = list(range(0, len(widths)-1, add_ori_interval))
            print('add_ori at layer:', self.add_ori)
        else:
            self.add_ori = []
        self.stack_output = stack_output
        self.input_transform = input_transform

        square_layers = []
        if square_num != 0 and square_ratio != 0:
            import numpy as np
            if square_layer_strategy == 0:
                square_layers = np.linspace(0, len(widths)-2, square_num+2)[1:-1]
            elif square_layer_strategy == 1:
                square_layers = np.linspace(0, len(widths)-2, square_num+1)[:-1]
            elif square_layer_strategy == 2:
                square_layers = np.linspace(0, len(widths)-2, square_num+1)[1:]
            square_layers = [int(i) for i in square_layers]
            print('square layers: ', square_layers)

        if f_lists is None:
            f_lists = [(8, 8)] * len(widths)
        assert len(f_lists) == len(widths)
        for i in range(len(widths) - 1):
            ratio = square_ratio if i in square_layers else 0
            self.append_layer(widths[i], widths[i + 1], f_lists[i], f_lists[i + 1],
                              square_before_relu=square_before_relu, square_ratio=ratio, add_ori=(i in self.add_ori))
        if self.stack_output:
            self.stack_1x1 = nn.ModuleList([nn.Conv2d(fs, widths[-1], 1) for fs in widths[1:]])

    def append_layer(self, in_channels, out_channels, previous_f_size, current_f_size, kernel_size=3, square_ratio=0,
                     square_before_relu=False, add_ori=False):
        if add_ori:
            in_channels += 3
        if self.layerType.startswith('normal'):
            new_layers = []
            if previous_f_size == current_f_size:
                new_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2))
            else:
                stride_w = previous_f_size[0] // current_f_size[0]
                stride_h = previous_f_size[1] // current_f_size[1]
                new_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                              stride=(stride_w, stride_h)))
            if add_ori:
                new_layers[0].weight.data[:, :3] *= (in_channels-3)/3
            if 'no_bn' not in self.layerType:
                new_layers.append(nn.BatchNorm2d(out_channels))

            if square_ratio != 0 and square_before_relu:
                new_layers.append(SquareLayer(square_ratio))

            if 'prelu' in self.layerType:
                new_layers.append(nn.PReLU())
            elif 'lrelu' in self.layerType:
                new_layers.append(nn.LeakyReLU())
            else:
                new_layers.append(nn.ReLU())

            if square_ratio !=0 and not square_before_relu:
                new_layers.append(SquareLayer(square_ratio))

            new_layer = nn.Sequential(*new_layers)
        elif self.layerType == 'repvgg':
            from model.basic_cifar_models.repvgg import RepVGGBlock
            stride_w = previous_f_size[0] // current_f_size[0]
            stride_h = previous_f_size[1] // current_f_size[1]
            new_layer = RepVGGBlock(in_channels, out_channels, kernel_size, stride=(stride_w, stride_h),
                                    padding=kernel_size // 2)
        elif self.layerType.startswith('plain_sr'):
            from frameworks.distillation.exp_network import Plain_SR_Block
            stride_w = previous_f_size[0] // current_f_size[0]
            stride_h = previous_f_size[1] // current_f_size[1]
            config = ""
            if '-' in self.layerType:
                config = self.layerType.split('-')[1]
            new_layer = Plain_SR_Block(in_channels, out_channels, kernel_size, stride=(stride_w, stride_h),
                                       padding=kernel_size // 2, config=config)
        else:
            raise NotImplementedError()
        self.append(new_layer)

    def append_tail(self, tailModule: nn.Module):
        import copy
        self.append(copy.deepcopy(tailModule))

    def forward(self, x, with_feature=False, start_forward_from=0, until=None, ori=None):
        f_list = []
        if self.input_transform is not None:
            x = self.input_transform(x)

        if len(self.add_ori) != 0 and ori is None:
            ori = x

        if self.stack_output:
            stack_out = 0
        else:
            stack_out = None

        ids = list(range(len(self)))
        for idx in ids[start_forward_from: until]:
            m = self[idx]

            # 尾部的 TailModule 直接接到输出
            if idx == len(self) - 1:
                if self.stack_output:
                    x = m(stack_out)
                else:
                    x = m(x)
            else:
                if idx in self.add_ori:
                    x = torch.cat([x, ori], dim=1)
                x = m(x)
                if self.stack_output:
                    stack_out = self.stack_1x1[idx](x) + stack_out

            if with_feature:
                f_list.append(x)

        return (f_list, x) if with_feature else x


class SquareLayer(nn.Module):
    def __init__(self, square_ratio):
        super().__init__()
        assert 0 < square_ratio < 1
        self.square_ratio = square_ratio

    def forward(self, x):
        c = int(x.size(1) * self.square_ratio)
        if c == 0:
            return x
        L = x[:, :c]**2
        R = x[:, c:]
        return torch.cat([L, R], dim=1)


class EasyScale(nn.Module):
    def __init__(self, scale, output_transform=None):
        super().__init__()
        self.up = nn.PixelShuffle(scale)
        self.out_transform = output_transform if output_transform is not None else nn.Identity()

    def forward(self, x):
        return self.out_transform(self.up(x))
