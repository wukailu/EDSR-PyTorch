# Distill Everything Into a Plain model
from typing import List

import torch
from torch import nn

from frameworks.distillation.feature_distillation import get_distill_module
from frameworks.lightning_base_model import LightningModule, _Module
from model import freeze, unfreeze_BN, freeze_BN
from model.basic_cifar_models.resnet_layerwise_cifar import LastLinearLayer
from model.layerwise_model import ConvertibleLayer, ConvertibleModel, pad_const_channel, ConvLayer, IdLayer, \
    merge_1x1_and_3x3


# TODO: fix conflicts between add_ori and init_stu_with_teacher

class DEIP_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.M_maps = []
        self.teacher_model = self.load_teacher()
        freeze(self.teacher_model.eval())

        self.plain_model = nn.ModuleList()

        import time
        start_time = time.process_time()
        self.init_student()
        print("initialization student width used ", time.process_time() - start_time)

        if self.params['add_ori'] and self.params['task'] == 'super-resolution':
            import copy
            self.sub_mean = copy.deepcopy(self.teacher_model.sequential_models[0].sub_mean)
        else:
            self.sub_mean = None

    def on_train_start(self):
        self.sync_plain_model()

    def sync_plain_model(self):  # this function can only be called after __init__ finished
        if self.trainer.num_gpus > 1:
            import torch.distributed as dist
            device = self.device
            TEMP_PATH = "temp_model_before_start.pt"
            if self.global_rank == 0:
                self.plain_model.cpu()
                torch.save(self.plain_model.state_dict(), TEMP_PATH)
            dist.barrier()
            self.plain_model.load_state_dict(torch.load(TEMP_PATH))
            self.plain_model.to(device)

    def load_teacher(self):
        teacher = _Module.load_from_checkpoint(checkpoint_path=self.params['teacher_pretrain_path']).model
        assert isinstance(teacher, ConvertibleModel)
        return ConvertibleModel(teacher.to_convertible_layers())

    def init_student(self):
        # First version, no progressive learning
        images = torch.stack([self.unpack_batch(self.dataProvider.train_dl.dataset[i])[0] for i in range(16)], dim=0)
        widths = [self.params['input_channel']] + self.calc_width(images=images)

        if self.params['task'] == 'super-resolution':
            for i in range(1, len(widths)):  # TODO: fix this bug in a better way
                widths[i] = min(widths[i - 1] * 9, widths[i])

        with torch.no_grad():
            f_list, _ = self.teacher_model(images, with_feature=True)
            f_shapes = [images.shape] + [f.shape for f in f_list]
            for idx in range(len(widths) - 2):
                self.append_layer(widths[idx], widths[idx + 1], f_shapes[idx][2:], f_shapes[idx + 1][2:])
        self.append_tail(widths[-2], widths[-1])

        if self.params['init_stu_with_teacher']:
            print("init student with teacher!")
            # teacher and student has different #input_channel and #out_channel
            # Method 1: ignore the relu layer, then student can be initialized at start
            # Method 1, branch 2: merge bias into conv by adding a constant channel to feature map
            # Method 2: Feature map mapping is kept by layer distillation, use the matrix from distillation, but this
            # need to be calculated on-the-fly.
            # Method 3: Feature map is sampled from teacher, and use decomposition technic to determined width and mapping,
            # generate student kernel with mapping and teacher kernel each layer.
            # Note 1: Pay attention to gradient vanishing after initialization.

            assert len(self.plain_model) == len(self.teacher_model.sequential_models)
            # Implement Method 1:
            M = torch.eye(3)
            for layer_s, layer_t in zip(self.plain_model[:-1], self.teacher_model.sequential_models[:-1]):
                M = self.init_layer(layer_s, layer_t, M)
                self.M_maps.append(M.detach())

            # LastFCLayer
            self.teacher_model.sequential_models[-1].init_student(self.plain_model[-1], M)

    def init_layer(self, layer_s, layer_t, M):  # M is of shape C_t x C_s
        assert isinstance(layer_t, ConvertibleLayer)

        if 'normal' in self.params['layer_type']:
            M = layer_t.init_student(layer_s, M)
            return M
        elif 'plain_sr' in self.params['layer_type']:
            M = layer_t.init_student(layer_s.conv, M)
            if layer_s.skip:
                k = layer_s.conv.kernel_size[0]
                for i in range(layer_s.conv.out_channels):
                    layer_s.conv.weight.data[i, i, k // 2, k // 2] -= 1
            return M
        elif self.params['layer_type'] == 'repvgg':
            from model.basic_cifar_models.repvgg import RepVGGBlock
            assert isinstance(layer_s, RepVGGBlock)
            conv_s: nn.Conv2d = layer_s.rbr_dense.conv
            conv = nn.Conv2d(in_channels=conv_s.in_channels, out_channels=conv_s.out_channels,
                             kernel_size=conv_s.kernel_size, stride=conv_s.stride, padding=conv_s.padding,
                             groups=conv_s.groups, bias=True)

            M = layer_t.init_student(conv, M)

            k = conv.kernel_size[0]
            if hasattr(layer_s, 'rbr_1x1'):
                for i in range(conv.out_channels):
                    for j in range(conv.in_channels):
                        conv.weight.data[i, j, k // 2, k // 2] -= layer_s.rbr_1x1.conv.weight.data[i, j, 0, 0]
            if hasattr(layer_s, 'rbr_identity'):
                for i in range(conv.out_channels):
                    conv.weight.data[i, i, k // 2, k // 2] -= 1

            layer_s.rbr_dense.bn.bias.data = conv.bias  # maybe we should average this into three part of bn
            layer_s.rbr_dense.conv.weight.data = conv.weight
            return M
        else:
            raise NotImplementedError()

    def complete_hparams(self):
        default_sr_list = {
            'task': 'classification',
            'input_channel': 3,
            'progressive_distillation': False,
            'init_with_teacher_param': False,
            'rank_eps': 5e-2,
            'use_bn': True,
            'add_ori': False,
            'layer_type': 'normal',
            'init_stu_with_teacher': False,
            'teacher_pretrain_path': None,
            'init_tail': False,
        }
        self.params = {**default_sr_list, **self.params}
        LightningModule.complete_hparams(self)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        if self.params['add_ori'] and self.hparams['task'] == 'classification':
            ori = x
        elif self.params['add_ori'] and self.hparams['task'] == 'super-resolution':
            ori = self.sub_mean(x)
        elif self.params['add_ori']:
            raise NotImplementedError()
        else:
            ori = None

        for m in self.plain_model[start_forward_from: until]:
            if self.params['add_ori']:
                x = torch.cat([x, ori], dim=1)
            x = m(pad_const_channel(x))
            if with_feature:
                f_list.append(x)
        return (f_list, x) if with_feature else x

    def unpack_batch(self, batch):
        if self.params['task'] == 'classification':
            images, labels = batch
        elif self.params['task'] == 'super-resolution':
            images, labels, filenames = batch
        else:
            raise NotImplementedError()
        return images, labels

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss

    def append_layer(self, in_channels, out_channels, previous_f_size, current_f_size, kernel_size=3):
        if self.params['add_ori']:
            in_channels += 3
        if self.params['layer_type'].startswith('normal'):
            if 'prelu' in self.params['layer_type']:
                act = nn.PReLU()
            else:
                act = nn.ReLU()
            bn = 'no_bn' not in self.params['layer_type']
            if previous_f_size == current_f_size:
                stride = 1
            else:
                stride_w = previous_f_size[0] // current_f_size[0]
                stride_h = previous_f_size[1] // current_f_size[1]
                stride = (stride_w, stride_h)

            new_layer = ConvLayer(in_channels, out_channels, kernel_size, bn=bn, act=act, stride=stride)
        elif self.params['layer_type'] == 'repvgg':
            # TODO: convert this to convertible layers
            from model.basic_cifar_models.repvgg import RepVGGBlock
            stride_w = previous_f_size[0] // current_f_size[0]
            stride_h = previous_f_size[1] // current_f_size[1]
            new_layer = RepVGGBlock(in_channels, out_channels, kernel_size, stride=(stride_w, stride_h),
                                    padding=kernel_size // 2)
        elif self.params['layer_type'].startswith('plain_sr'):
            # TODO: convert this to convertible layers
            from frameworks.distillation.exp_network import Plain_SR_Block
            stride_w = previous_f_size[0] // current_f_size[0]
            stride_h = previous_f_size[1] // current_f_size[1]
            config = ""
            if '-' in self.params['layer_type']:
                config = self.params['layer_type'].split('-')[1]
            new_layer = Plain_SR_Block(in_channels, out_channels, kernel_size, stride=(stride_w, stride_h),
                                       padding=kernel_size // 2, config=config)
        else:
            raise NotImplementedError()

        self.plain_model.append(new_layer)

    def append_tail(self, last_channel, output_channel):
        if self.params['task'] == 'classification':
            self.plain_model.append(LastLinearLayer(last_channel, output_channel))
        elif self.params['task'] == 'super-resolution':
            from model.super_resolution_model.edsr_layerwise_model import EDSRTail
            if isinstance(self.teacher_model.sequential_models[-1], EDSRTail):
                self.plain_model.append(
                    EDSRTail(self.params['scale'], last_channel, output_channel, 3, self.params['rgb_range']))
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def calc_width(self, images):
        ret = []
        with torch.no_grad():
            f_list, _ = self.teacher_model(images[:16], with_feature=True)
            teacher_width = [f.size(1) for f in f_list]
            if self.params['init_with_teacher_param']:  # calc width with model parameters instead of feature maps
                for m in self.teacher_model[:-2]:
                    assert isinstance(m, ConvertibleLayer)
                    mat = m.simplify_layer()[0].weight.detach().flatten(start_dim=1)
                    ret.append(rank_estimate(mat, eps=self.params['rank_eps']))
            else:
                for f in f_list[:-2]:
                    mat = f.transpose(0, 1).flatten(start_dim=1)
                    ret.append(rank_estimate(mat, eps=self.params['rank_eps']))
        ret.append(teacher_width[-2])
        ret.append(teacher_width[-1])  # num classes
        print("calculated teacher width = ", teacher_width)
        print("calculated student width = ", ret)
        return ret


class DEIP_Init(DEIP_LightModel):
    """
    全新的初始化方式，根据 feature map 采样决定各层的 feature map 映射以及学生网络的参数
    假设上一层 f_t = M f_s + b
    下一层 f'_t = M' f'_s + b'
    老师网络该层权重为 w_t, 对应 Conv3x3_t, 需推导出学生网络权重
    f'_t = Conv3x3_t(f_t) = Conv3x3_t(conv1x1_i(f_s)) = Conv3x3'(f_s)
    f'_t = M' Conv3x3_s(f_s) + b' = Conv1x1_{i+1}(Conv3x3_s(f_s))
    Conv3x3_s = Conv1x1_{i+1}^{-1} Conv3x3'(f_s)  ---> 等价于 解 线性方程组
    """

    def init_student(self):
        assert not self.params['add_ori']
        images = torch.stack([self.unpack_batch(self.dataProvider.train_dl.dataset[i])[0] for i in range(16)], dim=0)
        widths = [self.params['input_channel']]
        self.bridges = nn.ModuleList([IdLayer(self.params['input_channel'])])
        with torch.no_grad():
            f_list, _ = self.teacher_model(images[:16], with_feature=True)
            teacher_width = [f.size(1) for f in f_list]
            for f in f_list[:-2]:
                mat = f.transpose(0, 1).flatten(start_dim=1)
                # M*fs + bias \approx mat
                M, fs, bias, r = rank_estimate(mat, eps=self.params['rank_eps'], with_bias=True, with_rank=True,
                                               with_solution=True, use_NMF=False)
                print('fs_shape', fs.shape, 'fs_min', fs.min(), 'fs_mean', fs.mean())
                conv1x1 = nn.Conv2d(fs.size(0), mat.size(0), kernel_size=1, bias=True)
                conv1x1.weight.data[:] = M.reshape_as(conv1x1.weight)
                conv1x1.bias.data[:] = bias.reshape_as(conv1x1.bias)
                self.bridges.append(ConvLayer.fromConv2D(conv1x1))
                widths.append(r)
        widths.append(teacher_width[-2])
        self.bridges.append(IdLayer(teacher_width[-2]))
        widths.append(teacher_width[-1])
        self.bridges.append(IdLayer(teacher_width[-1]))
        print("calculated teacher width = ", [self.params['input_channel']] + teacher_width)
        print("calculated student width = ", widths)

        with torch.no_grad():
            for i in range(len(self.bridges) - 2):
                eq_conv, act = merge_1x1_and_3x3(self.bridges[i], self.teacher_model[i]).simplify_layer()
                conv = nn.Conv2d(widths[i] + 1, widths[i + 1], kernel_size=eq_conv.kernel_size, stride=eq_conv.stride,
                                 padding=eq_conv.padding, bias=False)
                if self.params['init_stu_with_teacher']:
                    print('Initializing layer...')
                    B = eq_conv.weight.data
                    M = self.bridges[i + 1].simplify_layer()[0].weight.data.flatten(start_dim=1)
                    M, bias = M[:, 1:], M[:, 0]
                    B[:, 0, B.size(2) // 2, B.size(3) // 2] -= bias
                    B = B.flatten(start_dim=1)
                    # solve MX=B
                    X = torch.lstsq(B, M)[0][:M.size(1)]
                    conv.weight.data[:] = X.reshape_as(conv.weight)
                self.plain_model.append(ConvLayer.fromConv2D(conv, act=act, const_channel_0=True))
        self.append_tail(widths[-2], widths[-1])
        if self.params['init_stu_with_teacher'] or self.params['init_tail']:
            self.teacher_model.sequential_models[-1].init_student(self.plain_model[-1], torch.eye(widths[-2]))


def test_rank(r, use_NMF, M, f2, f, with_solution, with_bias, with_rank, bias, eps, ret_err=False):
    ret = []
    if use_NMF:
        f = f[:, :512]
        from utils.NMF.snmf import SNMF
        method = SNMF(f.cpu().numpy(), num_bases=r)
        method.factorize(niter=200)
        app_f = torch.from_numpy(method.H)
        app_M = torch.from_numpy(method.W)
    else:
        app_M = M[:, :r]
        app_f = f2[:r]

    approx = torch.mm(app_M, app_f)
    error = torch.norm(f - approx, p=2) / torch.norm(f, p=2)
    # add relative eps
    # if error < eps * torch.max(torch.abs(f)) or error < eps:
    # if ((torch.abs(f - approx) / torch.max(f.abs(), torch.ones_like(f) * max_value * 0.1)) < eps).all():
    if error < eps:
        ret.append(True)
    else:
        ret.append(False)
    if with_solution:
        ret.append(app_M)
        ret.append(app_f)
    if with_bias:
        ret.append(bias)
    if with_rank:
        ret.append(r)
    if not ret_err:
        return ret
    else:
        return ret, error


def rank_estimate(f, eps=5e-2, with_rank=True, with_bias=False, with_solution=False, use_NMF=False):
    # TODO: consider how can we align f to 1-var or 1-norm
    """
    Estimate the size of feature map to approximate this. The return matrix f' should be positive if possible
    :param use_NMF: whether use NMF instead of SVD
    :param with_rank: shall we return rank of f'
    :param with_solution: shall we return f = M f'
    :param with_bias: whether normalize f to zero-mean
    :param f: tensor of shape (C, N)
    :param eps: the error bar for low_rank approximation
    """
    #  Here Simple SVD is used, which is the best approximation to min_{D'} ||D-D'||_F where rank(D') <= r
    #  A question is, how to solve min_{D'} ||(D-D')*W||_F where rank(D') <= r,
    #  W is matrix with positive weights and * is element-wise production
    #  refer to wiki, it's called `Weighted low-rank approximation problems`, which does not have an analytic solution
    assert len(f.shape) == 2
    # svd can not solve too large feature maps, so take samples
    if f.size(1) > 32768:
        perm = torch.randperm(f.size(1))[:32768]
        f = f[:, perm]

    if with_bias:
        bias = f.mean(dim=1, keepdim=True)
        f -= bias
    else:
        bias = torch.zeros_like(f).mean(dim=1, keepdim=True)

    if not use_NMF:
        u, s, v = torch.svd(f)
        M = u
        f2 = torch.mm(torch.diag(s), v.t())
    else:
        M, f2 = None, None

    final_ret = []
    L, R = 0, f.size(0)  # 好吧，不得不写倍增 [ )
    step = 1
    while L + step < R:
        ret = test_rank(L + step, use_NMF, M, f2, f, with_solution, with_bias, with_rank, bias, eps)
        if ret[0]:
            R = L + step
            final_ret = ret
            break
        else:
            step *= 2
    L = step // 2
    step = step // 2
    while step != 0:
        if L + step < R:
            ret = test_rank(L + step, use_NMF, M, f2, f, with_solution, with_bias, with_rank, bias, eps)
            if not ret[0]:
                L = L + step
            else:
                R = L + step
                final_ret = ret
        step = step // 2

    if len(final_ret) == 0:
        final_ret, error = test_rank(R, use_NMF, M, f2, f, with_solution, with_bias, with_rank, bias, eps, ret_err=True)
        print("rank estimation failed! feature shape is ", f.shape)
        print("max value and min value in feature is ", f.max(), f.min())
        print(f"rank estimation failed! The last error is {error}")

    if len(final_ret) == 2:
        return final_ret[1]
    else:
        return final_ret[1:]


# TODO: 啥时候把蒸馏改成多继承
class DEIP_Distillation(DEIP_LightModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.dist_method = self.get_distillation_module()

    def get_distillation_module(self):
        sample, _ = self.unpack_batch(self.train_dataloader().dataset[0])
        sample = sample.unsqueeze(dim=0)
        with torch.no_grad():
            feat_t, out_t = self.teacher_model(sample, with_feature=True)
            feat_s, out_s = self(sample, with_feature=True)
            dist_method = get_distill_module(self.params['dist_method'])(feat_s, feat_t)
        return dist_method

    def complete_hparams(self):
        default_sr_list = {
            'dist_method': 'FD_Conv1x1_MSE',
            'distill_coe': 1,
        }
        self.params = {**default_sr_list, **self.params}
        DEIP_LightModel.complete_hparams(self)

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)

        if self.training:
            feat_s, predictions = self(images, with_feature=True)
            task_loss = self.criterion(predictions, labels)

            with torch.no_grad():
                feat_t, out_t = self.teacher_model(images, with_feature=True)
            assert len(feat_s) == len(feat_t)
            dist_loss = self.dist_method(feat_s, feat_t, self.current_epoch / self.params['num_epochs'])
            loss = task_loss + dist_loss * self.params['distill_coe']

            self.log('train/dist_loss', dist_loss, sync_dist=True)
            self.log('train/task_loss', task_loss, sync_dist=True)
        else:
            predictions = self.forward(images)
            loss = self.criterion(predictions, labels)
            teacher_pred = self.teacher_model(images)
            metric = self.metric(teacher_pred, labels)
            self.log(phase + '/' + 'teacher_' + self.params['metric'], metric, sync_dist=True)

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss


class DEIP_Progressive_Distillation(DEIP_Distillation):
    def __init__(self, hparams):
        import numpy as np
        super().__init__(hparams)
        self.current_layer = 0
        self.milestone_epochs = [int(i) for i in
                                 np.linspace(0, self.params['num_epochs'], len(self.plain_model) + 1)[1:-1]]
        print(f'there are totally {len(self.plain_model)} layers in student, milestones are {self.milestone_epochs}')
        self.bridges = self.init_bridges()
        if not self.params['freeze_teacher_bn']:
            freeze_BN(self.teacher_model)
        else:
            unfreeze_BN(self.teacher_model)

    def init_bridges(self):
        sample, _ = self.unpack_batch(self.train_dataloader().dataset[0])
        sample = sample.unsqueeze(dim=0)
        ret = nn.ModuleList()
        with torch.no_grad():
            feat_t, out_t = self.teacher_model(sample, with_feature=True)
            feat_s, out_s = self(sample, with_feature=True)
            for fs, ft in zip(feat_s[:-1], feat_t[:-1]):
                ret.append(nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1, bias=False))
            ret.append(nn.Identity())
        return ret

    def complete_hparams(self):
        default_sr_list = {
            'freeze_trained': False,
            'freeze_teacher_bn': False,
        }
        self.params = {**default_sr_list, **self.params}
        DEIP_Distillation.complete_hparams(self)

    def on_train_epoch_start(self):
        if self.current_epoch in self.milestone_epochs:
            if self.params['freeze_trained']:
                print(f'freezing layer {self.current_layer}')
                freeze(self.plain_model[self.current_layer])  # use freeze will lead to large performance drop
            self.current_layer += 1
            self.milestone_epochs = self.milestone_epochs[1:]

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)

        if self.training:
            feat_s, predictions = self(images, with_feature=True)
            with torch.no_grad():
                feat_t, out_t = self.teacher_model(images, with_feature=True)
            assert len(feat_s) == len(feat_t)
            dist_loss = self.dist_method(feat_s, feat_t, self.current_epoch / self.params['num_epochs'])

            mid_feature = self.forward(images, until=self.current_layer + 1)
            transfer_feature = self.bridges[self.current_layer](mid_feature)
            predictions = self.teacher_model(transfer_feature, start_forward_from=self.current_layer + 1)
            task_loss = self.criterion(predictions, labels)
            loss = task_loss + dist_loss * self.params['distill_coe']

            self.log('train/dist_loss', dist_loss, sync_dist=True)
            self.log('train/task_loss', task_loss, sync_dist=True)
        else:
            mid_feature = self.forward(images, until=self.current_layer + 1)
            transfer_feature = self.bridges[self.current_layer](mid_feature)
            predictions = self.teacher_model(transfer_feature, start_forward_from=self.current_layer + 1)
            loss = self.criterion(predictions, labels)

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss


# 1. 逐层初始化学生，并且矩阵 M 来自于上一层的蒸馏结果
# 2. 统一矩阵 M 和蒸馏之间 Bridge, 增加一个常数层
# 3. 如何稳定训练，降低已训练层的学习率
class DEIP_Full_Progressive(DEIP_Progressive_Distillation):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.init_layer(self.plain_model[0], self.teacher_model.sequential_models[0], torch.eye(3))
        assert len(self.plain_model) == len(self.teacher_model.sequential_models)

    def on_train_epoch_start(self):
        from utils.tools import get_model_weight_hash
        print(f'plain model weight hash {get_model_weight_hash(self.plain_model)}')
        if self.current_epoch in self.milestone_epochs:
            with torch.no_grad():
                print(f'initializing layer {self.current_layer + 1}')
                if self.params['freeze_trained']:
                    print(f'freezing layer {self.current_layer}')
                    freeze(self.plain_model[self.current_layer])  # use freeze will lead to large performance drop
                self.milestone_epochs = self.milestone_epochs[1:]
                device = self.device
                self.cpu()
                M = self.bridges[self.current_layer].weight.data  # C_t x C_s
                M = M.reshape(M.shape[:2])
                print(f'M has shape {M.shape}, len(self.plain_model) = {len(self.plain_model)} ')
                self.current_layer += 1
                if self.current_layer != len(self.plain_model) - 1:
                    M = self.init_layer(self.plain_model[self.current_layer],
                                        self.teacher_model.sequential_models[self.current_layer], M)
                    self.bridges[self.current_layer].weight.data = M.reshape(list(M.shape) + [1, 1]).cuda()
                else:
                    self.teacher_model.sequential_models[-1].init_student(self.plain_model[-1], M)
                self.to(device)
            self.sync_plain_model()

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)

        mid_feature = self.forward(images, until=self.current_layer + 1)
        transfer_feature = self.bridges[self.current_layer](mid_feature)
        predictions = self.teacher_model(transfer_feature, start_forward_from=self.current_layer + 1)
        task_loss = self.criterion(predictions, labels)

        if self.training:
            loss = task_loss
            self.log('train/task_loss', task_loss, sync_dist=True)
        else:
            loss = task_loss

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss


def load_model(params):
    params = {'method': 'DirectTrain', **params}
    methods = {
        'DirectTrain': DEIP_LightModel,
        'Distillation': DEIP_Distillation,
        'Progressive_Distillation': DEIP_Progressive_Distillation,
        'DEIP_Full_Progressive': DEIP_Full_Progressive,
        'DEIP_Init': DEIP_Init,
    }
    print("using method ", params['method'])
    model = methods[params['method']](params)
    return model
