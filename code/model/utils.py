import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn, nn as nn
from torch.utils.tensorboard.summary import hparams
from datasets import query_dataset
from abc import ABC, abstractmethod


class Partial_Detach(nn.Module):
    def __init__(self, alpha=0):
        """
        When alpha = 0, it's completely detach, when alpha = 1, it's identity
        :param alpha:
        """
        super().__init__()
        self.alpha = 0

    def forward(self, inputs: torch.Tensor):
        if self.alpha == 0:
            return inputs.detach()
        elif self.alpha == 1:
            return inputs
        else:
            return inputs * self.alpha + inputs.detach() * (1 - self.alpha)


class Flatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor):
        return x.flatten(start_dim=self.start_dim)


def freeze(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_BN(model: torch.nn.Module):
    model.train()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = True


def freeze_BN(model: torch.nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            for p in m.parameters():
                p.requires_grad = False


def get_trainable_params(model):
    # print("Params to learn:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print("\t", repr(name))
            params_to_update.append(param)
    return params_to_update


def model_init(model: torch.nn.Module):
    from torch import nn
    for name, ch in model.named_children():
        print(f"{name} is initialized")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)) and m.weight.requires_grad:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MyTensorBoardLogger(TensorBoardLogger):

    def __init__(self, *args, **kwargs):
        super(MyTensorBoardLogger, self).__init__(*args, **kwargs)

    def log_hyperparams(self, *args, **kwargs):
        pass

    @rank_zero_only
    def log_hyperparams_metrics(self, params: dict, metrics: dict) -> None:
        params = self._convert_params(params)
        exp, ssi, sei = hparams(params, metrics)
        writer = self.experiment._get_file_writer()
        writer.add_summary(exp)
        writer.add_summary(ssi)
        writer.add_summary(sei)
        # some alternative should be added
        self.tags.update(params)


def get_classifier(classifier, dataset: str) -> torch.nn.Module:
    if isinstance(dataset, str):
        dataset_type = query_dataset(dataset)
    elif isinstance(dataset, dict):
        dataset_type = query_dataset(dataset['name'])
    else:
        raise TypeError("dataset must be either str or dict")
    num_classes = dataset_type.num_classes
    if isinstance(classifier, str):
        classifier_name = classifier
        params = {}
    elif isinstance(classifier, dict):
        classifier_name = classifier['arch']
        params = {key: value for key, value in classifier.items() if key != 'arch'}
    else:
        raise TypeError('Classifier should be either str or a dict with at least a key "arch".')

    classifier_name = classifier_name.lower()
    if classifier_name.startswith("Rep_"):
        from model.repdistiller_models import model_dict
        return model_dict[classifier_name[4:]](num_classes=num_classes, **params)
    elif classifier_name.endswith("_imagenet"):
        from model.imagenet_models import model_dict
        return model_dict[classifier_name[:-9]](num_classes=num_classes, **params)
    elif classifier_name.endswith("_sr"):
        from model.super_resolution_model import model_dict
        return model_dict[classifier_name[:-3]](num_classes=num_classes, **params)
    else:
        from model.basic_cifar_models import model_dict
        return model_dict[classifier_name](num_classes=num_classes, **params)


def load_models(hparams: dict) -> nn.ModuleList:
    num = len(hparams["pretrain_paths"])
    models: nn.ModuleList = nn.ModuleList([])
    for idx in range(num):
        if hparams["pretrain_paths"][idx].startswith("predefined_"):
            from model.basic_cifar_models import model_dict
            model = model_dict[hparams["pretrain_paths"][idx][len("predefined_"):]]()
        else:
            checkpoint = torch.load(hparams["pretrain_paths"][idx], map_location='cpu')
            try:
                # If it's a lightning model
                last_param = checkpoint['hyper_parameters']
                if 'dataset' in hparams and hparams['dataset'] != 'concat':
                    if last_param.dataset != hparams['dataset']:
                        print(
                            f"WARNING!!!!!!! Model trained on {last_param.dataset} will run on {hparams['dataset']}!!!!!!!")
                    assert query_dataset(last_param.dataset).num_classes == query_dataset(
                        hparams['dataset']).num_classes

                model = get_classifier(last_param.backbone, last_param.dataset)
                model.load_state_dict({key[6:]: value for key, value in checkpoint["state_dict"].items()})
            except RuntimeError as e:
                print("RuntimeError when loading models", e)
                model = get_classifier(hparams["classifiers"][idx], hparams["dataset"])
                model.load_state_dict(checkpoint["model"])
            except TypeError as e:
                print("TypeError when loading models", e)
                # Maybe it's just a torch.save(model) and torch.load(model)
                model = checkpoint
        models.append(model)
    return models


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'total number of params: {pytorch_total_params:,}')
    return pytorch_total_params


class LayerWiseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_models = nn.ModuleList()

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        for m in self.sequential_models[start_forward_from: until]:
            x = m(x)
            if with_feature:
                f_list.append(x)
        return (f_list, x) if with_feature else x

    def __len__(self):
        return len(self.sequential_models)


class ConvertibleLayer(nn.Module):

    def init_student(self, conv_s, M):
        conv = conv_to_no_bias_conv(self.simplify_layer()[0])
        return init_conv_with_conv(conv, conv_s, M)

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def simplify_layer(self):
        pass
        # return conv, ...


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


def conv_to_no_bias_conv(conv: nn.Conv2d):
    """
    convert a conv with bias to bias-free conv with a constant channel added to front of input data
    :param conv: a conv2d with input channel inc
    :return: a conv2d with input channel inc+1
    """
    if conv.bias is None:
        conv.bias = torch.zeros(conv.out_channels)
    ret = nn.Conv2d(conv.in_channels + 1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False,
                    padding_mode=conv.padding_mode)
    k_r, k_c = conv.kernel_size
    k_r, k_c = k_r // 2, k_c // 2
    ret.weight.data[:, 1:] = conv.weight.data[:, :]
    ret.weight.data[:, 0] = 0
    ret.weight.data[:, 0, k_r, k_c] = conv.bias.data
    return ret


def merge_1x1_and_3x3(conv1: nn.Conv2d, conv3: nn.Conv2d):
    """
    :param conv1 one conv2d of shape (out_1, in_1, 1, 1) with bias or not
    :param conv3 one conv2d of shape (out_2, out_1, k, k) with bias or not
    :return a conv2d of shape (out_2, in_1+1, k, k), where the input data should concat a channel full of 1 at data[:,0]
    """
    assert conv1.out_channels == conv3.in_channels
    assert conv1.stride == (1, 1)
    assert conv1.kernel_size == (1, 1)
    conv1 = conv_to_no_bias_conv(conv1)
    kernel = matmul_on_first_two_dim(conv3.weight.data, conv1.weight.data.view(conv1.weight.shape[:2]))
    if conv3.bias is not None:
        kernel[0, :, conv3.kernel_size[0] // 2, conv3.kernel_size[1] // 2] += conv3.bias
    conv = nn.Conv2d(in_channels=conv1.in_channels + 1, out_channels=conv3.out_channels, kernel_size=conv3.kernel_size,
                     stride=conv3.stride, padding=conv3.padding, bias=False)
    conv.weight.data = kernel
    return conv


def matmul_on_first_two_dim(m1: torch.Tensor, m2: torch.Tensor):
    """
    take matmul on first two dim only, and regard other dim as a scalar
    :param m1: tensor with at least 2 dim
    :param m2: tensor with at least 2 dim
    """
    if len(m1.shape) == 2:
        assert len(m2.shape) >= 2
        shape = m2.shape
        m2 = m2.flatten(start_dim=2).permute((2, 0, 1))
        ret = (m1 @ m2).permute((1, 2, 0))
        return ret.reshape(list(ret.shape[:2]) + list(shape[2:]))
    elif len(m2.shape) == 2:
        return matmul_on_first_two_dim(m2.transpose(0, 1), m1.transpose(0, 1)).transpose(0, 1)
    else:
        raise NotImplementedError()


def init_conv_with_conv(conv_t, conv_s, M):
    assert isinstance(conv_s, torch.nn.Conv2d)
    assert isinstance(conv_t, torch.nn.Conv2d)
    assert conv_s.stride == conv_t.stride
    assert conv_s.kernel_size == conv_t.kernel_size
    # 忽略Bias 误差 1e-5~1e-6, Bias = M^-1 Bias 误差 1e-2
    # 把 Kernel 看成一个 element 是向量的矩阵就行

    t_kernel = matmul_on_first_two_dim(conv_t.weight.data, M)

    u, s, v = torch.svd(t_kernel.flatten(start_dim=1))  # u and v are real orthogonal matrices
    r = conv_s.out_channels
    M = u[:, :r]

    s_kernel = (torch.diag(s[:r]) @ v.T[:r]).reshape(conv_s.weight.shape)
    conv_s.weight.data = s_kernel

    if conv_t.bias is None:
        conv_s.bias.data = 0
    elif conv_s.bias is not None:
        s_bias = M.pinverse() @ conv_t.bias.data
        conv_s.weight.bias = s_bias
    else:
        raise AttributeError("conv_s do not have bias while conv_t has bias, which is not possible to init s with t")
    return M
