from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn
from model import default_conv, matmul_on_first_two_dim, init_conv_with_conv, convbn_to_conv, SR_conv_init

"""
只有当所有 skip connection 分支上的值都是非负数时，才能转化为等宽的全卷积网络
否则需要增加一个常数层在输入上， 这里我们默认增加在 0 号 channel
"""


def conv_to_const_conv(conv: nn.Conv2d, add_input_channel=True):
    ret = nn.Conv2d(conv.in_channels if not add_input_channel else conv.in_channels + 1,
                    conv.out_channels, conv.kernel_size, padding=conv.padding, padding_mode=conv.padding_mode,
                    stride=conv.stride, bias=False)
    ret.weight.data[:] = 0
    if add_input_channel:
        ret.weight.data[:, 1:] = conv.weight.data
    else:
        ret.weight.data[:] = conv.weight.data
    if conv.bias is not None:
        ret.weight.data[:, 0, conv.kernel_size[0] // 2, conv.kernel_size[1] // 2] += conv.bias
    return ret


def pad_const_channel(x):
    ones = torch.ones_like(x[:, :1])
    return torch.cat([ones, x], dim=1)


class LayerWiseModel(nn.Module):
    def __init__(self, sequential_models=nn.ModuleList()):
        super().__init__()
        self.sequential_models = nn.ModuleList(sequential_models)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        for m in self.sequential_models[start_forward_from: until]:
            x = m(x)
            if with_feature:
                f_list.append(x)
        return (f_list, x) if with_feature else x

    def __len__(self):
        return len(self.sequential_models)

    def __getitem__(self, item):
        return self.sequential_models[item]

    def __iter__(self):
        return self.sequential_models.__iter__()


class ConvertibleModel(LayerWiseModel):
    """
    forward is normal forward
    sequential_models is list of Convertible layers
    remember to append const 1 to channel 0 for x, when calling forward for convertible layers
    usually the tail of this module is only initializable, not convertible
    """

    def to_convertible_layers(self):
        ret = []
        for m in self.sequential_models:
            if isinstance(m, InitializableLayer):
                ret.append(m)
            elif isinstance(m, ConvertibleModel):
                ret += m.to_convertible_layers()
            else:
                raise TypeError("Model can not be converted to plain model!")
        return simplify_sequential_model(ret)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        for m in self.sequential_models[start_forward_from: until]:
            x = m(pad_const_channel(x))
            if with_feature:
                f_list.append(x)
        return (f_list, x) if with_feature else x

    def __len__(self):
        ret = 0
        for m in self.sequential_models:
            if isinstance(m, InitializableLayer):
                ret += 1
            elif isinstance(m, ConvertibleModel):
                ret += len(m)
            else:
                raise TypeError("Model can not be converted to plain model!")
        return ret

    def generate_inference_model(self):
        ret = []
        layers = self.to_convertible_layers()
        for m in layers:
            if isinstance(m, ConvertibleLayer):
                ret.append(m.to_conv_layer())
            else:
                ret.append(m)
        return ConvertibleModel.from_convertible_models(ret)

    @staticmethod
    def from_convertible_models(model_list):
        return ConvertibleModel(nn.ModuleList(model_list))


class ConvertibleSubModel(ConvertibleModel):
    """
    和上面唯一的区别是，输入默认 x 已经还有常数层
    """

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        x = x[:, 1:]
        return ConvertibleModel.forward(self, x, with_feature, start_forward_from, until)


class SequentialConvertibleSubModel(ConvertibleSubModel):
    def __init__(self, *args):
        super().__init__()
        for m in args:
            assert isinstance(m, (ConvertibleLayer, ConvertibleSubModel))
            self.sequential_models.append(m)


class SkipConnectionSubModel(ConvertibleSubModel):
    def __init__(self, model_list, n_feats, skip_connection_bias=0, sum_output=True):
        super().__init__()
        self.model = SequentialConvertibleSubModel(*model_list)
        self.n_feats = n_feats
        self.bias = skip_connection_bias
        self.sum_output = sum_output

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        if with_feature:
            f_list, _ = self.model(x, with_feature, start_forward_from, until)
            f_list = [torch.cat([x[:, 1:], f], dim=1) for f in f_list]
            return f_list, f_list[-1]
        elif until is not None and until < len(self):
            ans = self.model(x, with_feature, start_forward_from, until)
            return torch.cat([x[:, 1:], ans], dim=1)
        else:
            ans = self.model(x, with_feature, start_forward_from, until)
            return x[:, 1:] + ans

    def __len__(self):
        return len(self.model)

    def to_convertible_layers(self):
        model_list = self.model.to_convertible_layers()
        assert len(model_list) >= 1
        if not isinstance(model_list[-1].simplify_layer()[1], nn.Identity):
            model_list.append(IdLayer(self.n_feats))

        ret = []
        id1 = IdLayer(self.n_feats, bias=self.bias, act=model_list[0].simplify_layer()[1])
        ret += [ConcatLayer(id1, model_list[0], share_input=True)]
        for m in model_list[1:-1]:
            ret += [ConcatLayer(IdLayer(self.n_feats, act=m.simplify_layer()[1]), m)]
        ret += [ConcatLayer(IdLayer(self.n_feats, bias=-self.bias), model_list[-1], sum_output=True)]
        return nn.ModuleList(ret)


class DenseFeatureFusionSubModel(ConvertibleSubModel):
    """
    把每个子模块的输出记录下来, concat 到输出最后面
    """
    def __init__(self, model_list, skip_connection_bias=0):
        super().__init__()
        self.models = nn.ModuleList(model_list)
        self.bias = skip_connection_bias

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        pass


class InitializableLayer(nn.Module):
    """
    forward 时 x 输入为 原本x concat 上全1的一层在 channel 0
    """

    def init_student(self, conv_s, M):
        """
        init student ConvLayer with teacher ConvertibleLayer
        :param conv_s: student ConvLayer
        :param M: matrix of shape C_t x C_s
        :return: new M
        """
        assert isinstance(conv_s, ConvLayer)
        conv, act = self.simplify_layer()
        conv_s.act = act
        M1 = torch.zeros((M.size(0) + 1, M.size(1) + 1))
        M1[0][0] = 1
        M1[1:, 1:] = M
        return init_conv_with_conv(conv, conv_s.conv, M1)


class ConvertibleLayer(InitializableLayer):
    """
    forward 时 x 输入为 原本x concat 上全1的一层在 channel 0
    simplify_layer 应该返回一个no bias卷积和act，卷积然后过 act，得到的结果应该和 forward 完全一致
    """

    @abstractmethod
    def forward(self, x):
        """
        a simple forward to get the right answer, here you can use any operation you want
        :param x: input data with const 1 at channel 0
        :return: forward result
        """
        pass

    @abstractmethod
    def simplify_layer(self) -> Tuple[nn.Conv2d, nn.Module]:
        """
        give a equivalent bias-less conv, act form of this layer
        act need to satisfy act(x)=x when x >= 0
        :return: conv, act, where output = act(conv(x))
        """
        pass

    def to_conv_layer(self):
        conv, act = self.simplify_layer()
        return ConvLayer.fromConv2D(conv, act, const_channel_0=True)


class ConvLayer(ConvertibleLayer):
    """
    stride is not supported so far
    """

    def __init__(self, in_channel, out_channel, kernel_size, stride=1, bn=False, act: nn.Module = nn.Identity(), SR_init=False):
        """
        create a Convertible Layer with a conv-bn-act structure, where the input has a const channel at 0.
        :param in_channel: original in_channels
        :param out_channel: original out_channels
        :param kernel_size: original kernel size
        :param stride: stride of conv
        :param bn: whether add bn
        :param act: what activation you want to use
        """
        super().__init__()
        self.conv = default_conv(in_channel + 1, out_channel, kernel_size, bias=False, stride=stride)
        if SR_init:
            SR_conv_init(self.conv)
        self.conv.weight.data[:, 0] = 0
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)
        else:
            self.bn = nn.Identity()
        self.act = act

    def simplify_layer(self):
        if isinstance(self.bn, nn.BatchNorm2d):
            return conv_to_const_conv(convbn_to_conv(self.conv, self.bn), add_input_channel=False), self.act
        else:
            return self.conv, self.act

    def forward(self, x):
        if isinstance(self.bn, nn.BatchNorm2d):
            return self.act(self.bn(self.conv(x)))
        else:
            return self.act(self.conv(x))

    @staticmethod
    def fromConv2D(conv: nn.Conv2d, act: nn.Module = nn.Identity(), const_channel_0=False):
        """
        build a ConvLayer from a normal nn.conv2d
        :param conv: nn.conv2d
        :param act: act after this conv, default to be identity
        :param const_channel_0: is this conv already take input channel 0 as a const channel with 1, default false
        :return:a ConvLayer
        """
        conv = conv_to_const_conv(conv, add_input_channel=not const_channel_0)
        ret = ConvLayer(conv.in_channels, conv.out_channels, conv.kernel_size, act=act)
        ret.conv = conv
        return ret


def merge_1x1_and_3x3(layer1, layer3):
    """
    :param layer1 ConvLayer of shape (out_1, in_1, 1, 1) with bias or not
    :param layer3 ConvLayer of shape (out_2, out_1, k, k) with bias or not
    :return a conv2d of shape (out_2, in_1, k, k), where the input data should concat a channel full of 1 at data[:,0]
    """
    assert isinstance(layer1, ConvertibleLayer) and isinstance(layer3, ConvertibleLayer)
    assert is_mergeable_1x1(layer1)
    conv1, act1 = layer1.simplify_layer()
    conv3, act3 = layer3.simplify_layer()
    assert conv1.out_channels + 1 == conv3.in_channels
    assert conv1.stride == (1, 1) and conv1.kernel_size == (1, 1)

    kernel = matmul_on_first_two_dim(conv3.weight.data[:, 1:], conv1.weight.data.view(conv1.weight.shape[:2]))
    kernel[:, 0] += conv3.weight.data[:, 0]
    conv = nn.Conv2d(in_channels=conv1.in_channels, out_channels=conv3.out_channels, kernel_size=conv3.kernel_size,
                     stride=conv3.stride, padding=conv3.padding, bias=False)
    conv.weight.data = kernel
    return ConvLayer.fromConv2D(conv, const_channel_0=True, act=act3)


def merge_3x3_and_1x1(layer3: ConvertibleLayer, layer1: ConvertibleLayer):
    # TODO: implement this
    assert isinstance(layer1, ConvertibleLayer)
    assert isinstance(layer3, ConvertibleLayer)
    assert NotImplementedError()
    pass


def pinv_1x1(layer1: ConvLayer):
    # TODO: implement this
    assert isinstance(pinv_1x1, ConvLayer)
    assert NotImplementedError()
    pass


class IdLayer(ConvertibleLayer):
    def __init__(self, channel, bias=0, act=nn.Identity()):
        super().__init__()
        self.channel = channel
        self.bias = bias
        self.act = act

    def simplify_layer(self):
        conv = ConvLayer(self.channel, self.channel, 1)
        conv.conv.weight.data[:, 0] = self.bias
        conv.conv.weight.data[:, 1:] = torch.eye(self.channel).view((self.channel, self.channel, 1, 1))
        return conv.conv, self.act

    def forward(self, x):
        return self.act(x[:, 1:] + self.bias)


def zero_pad(x, target_shape):
    """
    when x is of shape (a,b,c,d) and target shape is (e,f), x will be centered padded to (a,b,e,f)
    :param x: Tensor, input data that need to pad
    :param target_shape: target shape of last few dims
    :return: a padded tensor
    """
    pads = []
    for xs, ts in zip(x.shape[::-1], target_shape[::-1]):
        assert ts >= xs
        assert (ts - xs) % 2 == 0
        pads.append((ts - xs) // 2)
        pads.append((ts - xs) // 2)
    return torch.nn.functional.pad(x, pads)


class ConcatLayer(ConvertibleLayer):
    def __init__(self, layer1, layer2, share_input=False, sum_output=False):
        super().__init__()
        assert isinstance(layer1, ConvertibleLayer)
        assert isinstance(layer2, ConvertibleLayer)
        self.eq_conv1, act1 = layer1.simplify_layer()
        self.eq_conv2, act2 = layer2.simplify_layer()
        assert isinstance(act1, type(act2))
        self.act = act1
        self.share_input = share_input
        self.sum_output = sum_output

        assert self.eq_conv1.kernel_size[0] == self.eq_conv1.kernel_size[1]
        assert self.eq_conv2.kernel_size[0] == self.eq_conv2.kernel_size[1]
        assert self.eq_conv1.padding[0] == self.eq_conv1.padding[1]
        assert self.eq_conv1.kernel_size[0] // 2 == self.eq_conv1.padding[0]
        assert self.eq_conv2.padding[0] == self.eq_conv2.padding[1]
        assert self.eq_conv2.kernel_size[0] // 2 == self.eq_conv2.padding[0]
        assert self.eq_conv1.kernel_size[0] % 2 == 1
        assert self.eq_conv2.kernel_size[0] % 2 == 1
        assert self.eq_conv1.stride == self.eq_conv2.stride
        assert self.eq_conv1.padding_mode == self.eq_conv2.padding_mode

    def forward(self, x):
        if self.share_input:
            x1, x2 = x, x
        else:
            assert x.size(1) + 1 == self.eq_conv1.in_channels + self.eq_conv2.in_channels
            x1, x2 = x[:, :self.eq_conv1.in_channels], pad_const_channel(x[:, self.eq_conv1.in_channels:])

        x1, x2 = self.eq_conv1(x1), self.eq_conv2(x2)

        if self.sum_output:
            assert x1.shape == x2.shape
            ret = x1 + x2
        else:
            ret = torch.cat([x1, x2], dim=1)
        return self.act(ret)

    def simplify_layer(self):
        if self.share_input:
            assert self.eq_conv1.in_channels == self.eq_conv2.in_channels
            in_channels = self.eq_conv1.in_channels
        else:
            in_channels = self.eq_conv1.in_channels + self.eq_conv2.in_channels - 1

        if self.sum_output:
            assert self.eq_conv1.out_channels == self.eq_conv2.out_channels
            out_channels = self.eq_conv1.out_channels
        else:
            out_channels = self.eq_conv1.out_channels + self.eq_conv2.out_channels

        conv = nn.Conv2d(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=max(self.eq_conv1.kernel_size, self.eq_conv2.kernel_size),
                         padding=max(self.eq_conv1.padding, self.eq_conv2.padding),
                         padding_mode=self.eq_conv1.padding_mode,
                         bias=False)
        conv.weight.data[:] = 0

        bias1 = zero_pad(self.eq_conv1.weight.data[:, 0], conv.kernel_size)
        bias2 = zero_pad(self.eq_conv2.weight.data[:, 0], conv.kernel_size)
        if self.sum_output:
            conv.weight.data[:, 0] = bias1 + bias2
        else:
            conv.weight.data[:, 0] = torch.cat([bias1, bias2], dim=0)

        kernel = torch.zeros_like(conv.weight.data[:, 1:])
        kernel1 = zero_pad(self.eq_conv1.weight.data[:, 1:], conv.kernel_size)
        kernel2 = zero_pad(self.eq_conv2.weight.data[:, 1:], conv.kernel_size)

        slice_in = slice(None, kernel1.size(1)) if self.share_input else slice(kernel1.size(1), None)
        slice_out = slice(None, kernel1.size(0)) if self.sum_output else slice(kernel1.size(0), None)
        kernel[:kernel1.size(0), :kernel1.size(1)] += kernel1
        kernel[slice_out, slice_in] += kernel2
        conv.weight.data[:, 1:] = kernel
        return conv, self.act


def is_mergeable_1x1(conv):
    if conv is None or not isinstance(conv, ConvertibleLayer):
        return False
    conv, act = conv.simplify_layer()
    if conv.kernel_size == (1, 1) and isinstance(act, nn.Identity):
        return True
    return False


def simplify_sequential_model(model_list):
    ret = []
    pre_1x1 = None
    for m in model_list:
        if pre_1x1 is None:
            if is_mergeable_1x1(m):
                pre_1x1 = m
            else:
                ret += [m]
        else:
            pre_1x1 = merge_1x1_and_3x3(pre_1x1, m)
            if not is_mergeable_1x1(pre_1x1):
                ret += [pre_1x1]
                pre_1x1 = None
            else:
                pass
    if pre_1x1 is not None:
        ret += [pre_1x1]
    return nn.ModuleList(ret)
