import torch.nn as nn
from collections import OrderedDict
from .encoder import get_encoder
from .base import BaseNet

from model.super_resolution_model.plain_layerwise_model import Plain_layerwise_Model, EasyScale


class PlainNet(nn.Module):
    def __init__(self, in_nc=3, n_feats=50, num_modules=16, out_nc=3, scale=4, **kwargs):
        super().__init__()
        nf = n_feats
        widths = [in_nc] + [nf] * num_modules
        tailModule = EasyScale(scale)
        widths += [(scale ** 2) * out_nc]
        model = Plain_layerwise_Model(widths=widths, **kwargs)
        model.append_tail(tailModule)
        self.network = nn.Sequential(
            OrderedDict([
                ('feature_extraction', nn.Sequential(*model.sequential_models[:-1])),
                ('last_layer', nn.Sequential(*model.sequential_models[-1:])),
            ]))
        from model import model_init
        model_init(self.network)

    def forward(self, x):
        return self.network(x)


class PlainNetAutoencoder(nn.Module):
    def __init__(self, scale, n_colors, d=56, s=12, m=4, k=1, encoder='inv_fsrcnn', n_feats=50, num_modules=16):
        super().__init__()
        self.encoder = get_encoder(encoder, scale=scale, d=d, s=s, k=k, n_colors=n_colors)
        model = PlainNet(n_feats=n_feats, num_modules=num_modules, scale=scale, in_nc=n_colors, out_nc=n_colors)

        self.network = nn.Sequential(
            OrderedDict([
                ('encoder', nn.Sequential(*self.encoder)),
                ('feature_extraction', model.network[0]),
                ('last_layer', model.network[1]),
            ]))

    def forward(self, x):
        return self.network(x)


class PlainNetStudent(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m=4, k=1, vid_info=None, n_feats=50, num_modules=16,
                 modules_to_freeze=None, initialize_from=None,
                 modules_to_initialize=None):
        super().__init__()
        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_freeze = modules_to_freeze
        self.modules_to_initialize = modules_to_initialize
        self.backbone = PlainNet(n_feats=n_feats, num_modules=num_modules, scale=scale, in_nc=n_colors, out_nc=n_colors)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, LR, HR=None, teacher_pred_dict=None):
        ret_dict = dict()
        x = LR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](x)
                var = self.vid_module_dict._modules[layer_name + '_var'](x)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var
        hr = x
        ret_dict['hr'] = hr
        return ret_dict


class PlainNetTeacher(BaseNet):
    def __init__(self, scale, n_colors, d=56, s=12, m=4, k=1, vid_info=None, n_feats=50, num_modules=16,
                 modules_to_freeze=None, initialize_from=None, modules_to_initialize=None,
                 encoder='inv_fsrcnn'):
        super().__init__()

        self.scale = scale
        self.initialize_from = initialize_from
        self.modules_to_initialize = modules_to_initialize
        self.modules_to_freeze = modules_to_freeze
        self.backbone = PlainNetAutoencoder(scale, n_colors, d, s, m, k, encoder, n_feats, num_modules)
        self.vid_info = vid_info if vid_info is not None else []
        self.vid_module_dict = self.get_vid_module_dict()

        if initialize_from is not None:
            self.load_pretrained_model()
        if modules_to_freeze is not None:
            self.freeze_modules()

    def forward(self, HR, LR=None):
        ret_dict = dict()

        x = HR
        layer_names = self.backbone.network._modules.keys()
        for layer_name in layer_names:
            x = self.backbone.network._modules[layer_name](x)
            ret_dict[layer_name] = x
            if layer_name in self.distill_layers:
                mean = self.vid_module_dict._modules[layer_name + '_mean'](x)
                var = self.vid_module_dict._modules[layer_name + '_var'](x)
                ret_dict[layer_name + '_mean'] = mean
                ret_dict[layer_name + '_var'] = var

        hr = x
        ret_dict['hr'] = hr
        return ret_dict


def get_plainnet_teacher(scale, n_colors, **kwargs):
    return PlainNetTeacher(scale, n_colors, **kwargs)


def get_plainnet_student(scale, n_colors, **kwargs):
    return PlainNetStudent(scale, n_colors, **kwargs)
