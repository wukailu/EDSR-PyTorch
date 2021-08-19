import torch
import torch.nn as nn

from .utils import register_model
from .. import LayerWiseModel


@register_model
def Plain_layerwise(in_nc=3, n_feats=50, nf=None, num_modules=4, out_nc=3, scale=4, **kwargs):
    nf = n_feats if nf is None else nf
    widths = [in_nc] + [nf] * num_modules + [(scale ** 2) * out_nc]
    model = Plain_layerwise_Model(widths=widths, **kwargs)
    model.append_tail(EasyScale(scale))
    return model


class Plain_layerwise_Model(LayerWiseModel):
    def __init__(self, widths, layerType='normal_no_bn', input_transform=None, f_lists=None, add_ori=False,
                 stack_output=False, **kwargs):
        """
        :arg add_ori if this is true, there will be 3 more channel on input, which is original input data
        :arg stack_output if this is true, all feature maps will be stacked and pass by a 1x1 conv to generate output,
        otherwise output is just the result from last conv
        :type stack_output: bool
        :type add_ori: bool
        """
        super().__init__()
        self.layerType = layerType
        self.add_ori = add_ori
        self.stack_output = stack_output
        self.input_transform = input_transform
        if f_lists is None:
            f_lists = [(8, 8)] * len(widths)
        assert len(f_lists) == len(widths)
        for i in range(len(widths) - (2 if self.stack_output else 1)):
            self.append_layer(widths[i], widths[i + 1], f_lists[i], f_lists[i + 1])
        if self.stack_output:
            self.sequential_models.append(nn.Conv2d(sum(widths[1:-1]) + (3 if self.add_ori else 0), widths[-1], 1))

    def append_layer(self, in_channels, out_channels, previous_f_size, current_f_size, kernel_size=3):
        if self.add_ori:
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
            if 'no_bn' not in self.layerType:
                new_layers.append(nn.BatchNorm2d(out_channels))
            if 'prelu' in self.layerType:
                new_layers.append(nn.PReLU())
            elif 'lrelu' in self.layerType:
                new_layers.append(nn.LeakyReLU())
            else:
                new_layers.append(nn.ReLU())
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
        self.sequential_models.append(new_layer)

    def append_tail(self, tailModule: nn.Module):
        import copy
        self.sequential_models.append(copy.deepcopy(tailModule))

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        if self.input_transform is not None:
            x = self.input_transform(x)

        if self.add_ori:
            ori = x
        else:
            ori = None

        ids = list(range(len(self.sequential_models)))
        for idx in ids[start_forward_from: until]:
            m = self.sequential_models[idx]

            if idx == len(self.sequential_models) - 1:
                x = m(x)
            else:
                if self.add_ori:
                    x = torch.cat([x, ori], dim=1)
                if idx == len(self.sequential_models) - 2 and self.stack_output:
                    x = m(torch.cat(f_list[:-1] + [x], dim=1))
                else:
                    x = m(x)

            if with_feature or self.stack_output:
                f_list.append(x)

        return (f_list, x) if with_feature else x


class EasyScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.up = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.up(x)
