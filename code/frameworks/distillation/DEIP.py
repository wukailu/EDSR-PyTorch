# Distill Everything Into a Plain model
from typing import List

import torch
from torch import nn

from frameworks.distillation.feature_distillation import get_distill_module
from frameworks.lightning_base_model import LightningModule, _Module
from model import freeze, unfreeze_BN, freeze_BN, std_alignment
from model.basic_cifar_models.resnet_layerwise_cifar import LastLinearLayer
from model.layerwise_model import ConvertibleLayer, ConvertibleModel, pad_const_channel, ConvLayer, IdLayer, \
    merge_1x1_and_3x3


# TODO: fix conflicts between add_ori and init_stu_with_teacher

class DEIP_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.M_maps = []
        self.teacher_plain_model, self.teacher = self.load_teacher()
        freeze(self.teacher_plain_model.eval())

        self.plain_model = nn.ModuleList()
        self.fs_std = []
        self.example_data = torch.stack([self.unpack_batch(self.dataProvider.train_dl.dataset[i])[0] for i in range(16)], dim=0)

        import time
        start_time = time.process_time()
        self.init_student()
        print("initialization student width used ", time.process_time() - start_time)
        if self.params['std_align']:
            std_alignment(self.plain_model, self.example_data, self.fs_std)
            print("std alignment finished ")

        if self.params['add_ori'] and self.params['task'] == 'super-resolution':
            import copy
            self.sub_mean = copy.deepcopy(self.teacher_plain_model.sequential_models[0].sub_mean)
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
        return ConvertibleModel(teacher.to_convertible_layers()), teacher

    def init_student(self):
        # First version, no progressive learning
        images = torch.stack([self.unpack_batch(self.dataProvider.train_dl.dataset[i])[0] for i in range(16)], dim=0)
        widths = [self.params['input_channel']] + self.calc_width(images=images)

        if self.params['task'] == 'super-resolution':
            for i in range(1, len(widths)):  # TODO: fix this bug in a better way
                widths[i] = min(widths[i - 1] * 9, widths[i])

        with torch.no_grad():
            f_list, _ = self.teacher(images, with_feature=True)
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

            assert len(self.plain_model) == len(self.teacher_plain_model.sequential_models)
            # Implement Method 1:
            M = torch.eye(3)
            for layer_s, layer_t in zip(self.plain_model[:-1], self.teacher_plain_model.sequential_models[:-1]):
                M = self.init_layer(layer_s, layer_t, M)
                self.M_maps.append(M.detach())

            # LastFCLayer
            self.teacher_plain_model.sequential_models[-1].init_student(self.plain_model[-1], M)

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
            'std_align': False,
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
        assert len(previous_f_size) == 2
        assert len(current_f_size) == 2
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
                assert stride_h >= 1
                assert stride_w >= 1
                stride = (stride_w, stride_h)

            new_layer = ConvLayer(in_channels, out_channels, kernel_size, bn=bn, act=act, stride=stride,
                                  SR_init=self.params['task'] == 'super-resolution')
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
            from model.super_resolution_model.rdn_layerwise_model import RDN_Tail
            if isinstance(self.teacher_plain_model.sequential_models[-1], EDSRTail):
                self.plain_model.append(
                    EDSRTail(self.params['scale'], last_channel, output_channel, 3, self.params['rgb_range']))
            elif isinstance(self.teacher_plain_model.sequential_models[-1], RDN_Tail):
                self.plain_model.append(
                    RDN_Tail(last_channel, self.params['scale'], 3, 3, last_channel, remove_const_channel=True)
                )
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def calc_width(self, images):
        ret = []
        with torch.no_grad():
            f_list, _ = self.teacher(images[:16], with_feature=True)
            teacher_width = [f.size(1) for f in f_list]
            if self.params['init_with_teacher_param']:  # calc width with model parameters instead of feature maps
                for m in self.teacher_plain_model[:-2]:
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


class DEIP_Dropout_Init(DEIP_LightModel):
    """
    根据 feature map 决定学生的网络宽度，然后通过对老师的channel 进行 dropout 实现变窄。
    """

    def init_student(self):
        assert not self.params['add_ori']
        images = self.unpack_batch(next(iter(self.dataProvider.train_dl)))[0]
        widths = [self.params['input_channel']]
        with torch.no_grad():
            f_list, _ = self.teacher(images, with_feature=True)
            teacher_width = [f.size(1) for f in f_list]
            for f in f_list[:-2]:
                mat = f.transpose(0, 1).flatten(start_dim=1)
                # M*fs + bias \approx mat
                _, r = rank_estimate(mat, eps=self.params['rank_eps'], with_bias=True)
                widths.append(r)
        widths.append(teacher_width[-2])
        widths.append(teacher_width[-1])
        print("calculated teacher width = ", [self.params['input_channel']] + teacher_width)
        print("calculated student width = ", widths)

        with torch.no_grad():
            M = torch.arange(widths[0])
            for i in range(len(widths) - 2):
                eq_conv, act = self.teacher_plain_model[i].simplify_layer()
                conv = nn.Conv2d(widths[i] + 1, widths[i + 1], kernel_size=eq_conv.kernel_size, stride=eq_conv.stride,
                                 padding=eq_conv.padding, bias=False)
                if self.params['init_stu_with_teacher']:
                    print('Initializing layer...')
                    X = eq_conv.weight.data[:, 1:][:, M]
                    bias = eq_conv.weight.data[:, 0]
                    M = torch.sort(torch.randperm(X.size(0))[:widths[i + 1]])[0]
                    X = X[M] * (X.size(1) / widths[i])  # 向前 scale
                    bias = bias[M]
                    conv.weight.data[:, 1:] = X
                    conv.weight.data[:, 0] = bias
                self.plain_model.append(ConvLayer.fromConv2D(conv, act=act, const_channel_0=True))
        self.append_tail(widths[-2], widths[-1])
        if self.params['init_stu_with_teacher'] or self.params['init_tail']:
            self.teacher_plain_model.sequential_models[-1].init_student(self.plain_model[-1], torch.eye(widths[-2]))


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

    #  try to keep app_f and f has the same variance
    # scale = f.std(unbiased=False) / app_f.std(unbiased=False)
    # app_f *= scale
    # app_M /= scale

    approx = torch.mm(app_M, app_f)
    error = torch.norm(f - approx, p=2) / torch.norm(f, p=2)

    if with_bias:
        # adjust the app_f to most positive value
        neg = app_f.clone()
        neg[neg > 0] = 0
        adjust = -neg.mean(dim=1, keepdim=True) * 3
        app_f = app_f + adjust
        bias -= app_M @ adjust

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


def rank_estimate(f, eps=5e-2, with_rank=True, with_bias=False, with_solution=False, use_NMF=False, fix_r=-1):
    # TODO: consider how can we align f to 1-var or 1-norm
    """
    Estimate the size of feature map to approximate this. The return matrix f' should be positive if possible
    :param fix_r: just fix the returned width as r = fix_r to get the decomposition results
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

    if fix_r != -1:
        fix_r = min(fix_r, f.size(0))
        return test_rank(fix_r, use_NMF, M, f2, f, with_solution, with_bias, with_rank, bias, eps, ret_err=False)[1:]

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
        if self.params['fix_distill_module']:
            freeze(self.dist_method)

    def get_distillation_module(self):
        sample, _ = self.unpack_batch(self.train_dataloader().dataset[0])
        sample = sample.unsqueeze(dim=0)
        with torch.no_grad():
            feat_t, out_t = self.teacher(sample, with_feature=True)
            feat_s, out_s = self(sample, with_feature=True)
            distill_config = self.params['dist_method']
            dist_method = get_distill_module(distill_config['name'])(feat_s, feat_t, **distill_config)
        return dist_method

    def complete_hparams(self):
        default_sr_list = {
            'dist_method': 'FD_Conv1x1_MSE',
            'fix_distill_module': False,
            'distill_coe': 0,
            'distill_alpha': 1,
            'distill_coe_mod': 'old',
        }
        self.params = {**default_sr_list, **self.params}
        DEIP_LightModel.complete_hparams(self)
        if isinstance(self.params['dist_method'], str):
            self.params['dist_method'] = {'name': self.params['dist_method']}

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)

        if self.training:
            feat_s, predictions = self(images, with_feature=True)
            task_loss = self.criterion(predictions, labels)
            self.log('train/task_loss', task_loss, sync_dist=True)

            if self.params['distill_coe'] != 0:
                with torch.no_grad():
                    feat_t, out_t = self.teacher(images, with_feature=True)
                assert len(feat_s) == len(feat_t)
                ratio = self.current_epoch / self.params['num_epochs']
                dist_loss = self.dist_method(feat_s, feat_t, ratio)

                coe_task = 1
                coe_dist = self.params['distill_coe'] * (self.params['distill_alpha'] ** ratio)
                if self.params['distill_coe_mod'] != 'old':
                    coe_sum = (coe_task + coe_dist)
                    coe_task /= coe_sum
                    coe_dist /= coe_sum

                loss = task_loss * coe_task + dist_loss * coe_dist
                self.log('train/dist_loss', dist_loss, sync_dist=True)
            else:
                loss = task_loss
        else:
            predictions = self.forward(images)
            loss = self.criterion(predictions, labels)
            if self.params['distill_coe'] != 0:
                teacher_pred = self.teacher(images)
                metric = self.metric(teacher_pred, labels)
                self.log(phase + '/' + 'teacher_' + self.params['metric'], metric, sync_dist=True)

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric, sync_dist=True)
        return loss


class DEIP_Init(DEIP_Distillation):
    """
    全新的初始化方式，根据 feature map 采样决定各层的 feature map 映射以及学生网络的参数
    假设上一层 f_t = M f_s + b
    下一层 f'_t = M' f'_s + b'
    老师网络该层权重为 w_t, 对应 Conv3x3_t, 需推导出学生网络权重
    f'_t = Conv3x3_t(f_t) = Conv3x3_t(conv1x1_i(f_s)) = Conv3x3'(f_s)
    f'_t = M' Conv3x3_s(f_s) + b' = Conv1x1_{i+1}(Conv3x3_s(f_s))
    Conv3x3_s = Conv1x1_{i+1}^{-1} Conv3x3'(f_s)  ---> 等价于 解 线性方程组
    """

    def complete_hparams(self):
        default_sr_list = {
            'dist_method': 'BridgeDistill',
            'ridge_alpha': 0.1,
        }
        self.params = {**default_sr_list, **self.params}
        DEIP_Distillation.complete_hparams(self)

    def get_distillation_module(self):
        distill_config = self.params['dist_method']
        assert distill_config['name'] == 'BridgeDistill'
        from frameworks.distillation.feature_distillation import BridgeDistill
        return BridgeDistill(self.bridges[1:], **distill_config)

    def init_student(self):
        assert not self.params['add_ori']
        widths = [self.params['input_channel']]
        self.bridges = nn.ModuleList([IdLayer(self.params['input_channel'])])

        self.fs_his = []
        self.ft_his = []

        with torch.no_grad():
            f_list, _ = self.teacher(self.example_data, with_feature=True)
            self.ft_his = f_list
            teacher_width = [f.size(1) for f in f_list]
            for f in f_list[:-2]:
                mat = f.transpose(0, 1).flatten(start_dim=1)
                fix_r = self.params['fix_r'] if 'fix_r' in self.params else -1

                # M*fs + bias \approx mat
                M, fs, bias, r = rank_estimate(mat, eps=self.params['rank_eps'], with_bias=True, with_rank=True,
                                               with_solution=True, use_NMF=False, fix_r=fix_r)
                conv1x1 = nn.Conv2d(fs.size(0), mat.size(0), kernel_size=1, bias=True)
                conv1x1.weight.data[:] = M.reshape_as(conv1x1.weight)
                conv1x1.bias.data[:] = bias.reshape_as(conv1x1.bias)

                self.fs_std.append(fs.std())
                self.fs_his.append(fs)

                print('---------layer ', len(self.bridges), '--------')
                print('mat_shape', mat.shape, 'mat_min', mat.min(), 'mat_mean', mat.mean(), 'mat_std', f.std())
                print('ft_shape', f.shape, 'f_min', f.min(), 'f_mean', f.mean(), 'f_std', f.std())
                print('fs_shape', fs.shape, 'fs_min', fs.min(), 'fs_mean', fs.mean(), 'fs_std', fs.std())
                print('M_shape', M.shape, 'M_min', M.min(), 'M_mean', M.mean(), 'M_std', M.std())
                print('bias_shape', bias.shape, 'bias_min', bias.min(), 'bias_mean', bias.mean(), 'bias_std', bias.std())

                self.bridges.append(ConvLayer.fromConv2D(conv1x1))
                widths.append(r)
            self.fs_std.append(f_list[-2].std())
        widths.append(teacher_width[-2])
        self.bridges.append(IdLayer(teacher_width[-2]))
        widths.append(teacher_width[-1])
        self.bridges.append(IdLayer(teacher_width[-1]))
        print("calculated teacher width = ", [self.params['input_channel']] + teacher_width)
        print("calculated student width = ", widths)
        f_shapes = [self.example_data.shape[-2:]] + [f.shape[-2:] for f in f_list[:-1]]
        with torch.no_grad():
            for i in range(len(self.bridges) - 2):
                if self.params['init_stu_with_teacher']:
                    print('Initializing layer...')
                    eq_conv, act = merge_1x1_and_3x3(self.bridges[i], self.teacher_plain_model[i]).simplify_layer()
                    conv = nn.Conv2d(widths[i] + 1, widths[i + 1], kernel_size=eq_conv.kernel_size,
                                     stride=eq_conv.stride,
                                     padding=eq_conv.padding, bias=False)
                    B = eq_conv.weight.data
                    M = self.bridges[i + 1].simplify_layer()[0].weight.data.flatten(start_dim=1)
                    M, bias = M[:, 1:], M[:, 0]
                    B[:, 0, B.size(2) // 2, B.size(3) // 2] -= bias
                    B = B.flatten(start_dim=1)
                    # solve MX=B
                    # example, M: 20x12, X: 12x144, B:20x144
                    # method 1: might got super large single value in X
                    # X = torch.lstsq(B, M)[0][:M.size(1)]
                    # method 2: no bias included
                    from sklearn.linear_model import Ridge
                    clf = Ridge(alpha=self.params['ridge_alpha'], fit_intercept=False)
                    clf.fit(M.numpy(), B.numpy())
                    X = torch.tensor(clf.coef_.T)
                    # method 3: bias included, MX = B where B is centered, this method has problem that B is a kernel,
                    # normalize a kernel will introduce problem
                    # from sklearn.linear_model import Ridge
                    # clf = Ridge(alpha=0.01, fit_intercept=False)
                    # n_bias = B.mean(dim=1).reshape((-1, 1))
                    # B -= n_bias
                    # clf.fit(M.numpy(), B.numpy())
                    # X = torch.tensor(clf.coef_.T)
                    # if isinstance(self.bridges[i+1], IdLayer):
                    #     self.bridges[i+1].bias += n_bias
                    # elif isinstance(self.bridges[i+1], ConvLayer):
                    #     self.bridges[i+1].conv.weight.data[:, 0, 0, 0] += n_bias[:, 0]
                    # else:
                    #     raise NotImplementedError()
                    print('B_mean', B.mean(), 'B_std', B.std(), 'B_max', B.max(), 'B_min', B.min())
                    print('M_mean', M.mean(), 'M_std', M.std(), 'M_max', M.max(), 'M_min', M.min())
                    print('X_mean', X.mean(), 'X_std', X.std(), 'X_max', X.max(), 'X_min', X.min())

                    conv.weight.data[:] = X.reshape_as(conv.weight)
                    self.plain_model.append(ConvLayer.fromConv2D(conv, act=act, const_channel_0=True, version=self.params['layer_type']))
                else:
                    self.append_layer(widths[i], widths[i + 1], f_shapes[i], f_shapes[i + 1])
        self.append_tail(widths[-2], widths[-1])
        if self.params['init_stu_with_teacher'] or self.params['init_tail']:
            self.teacher_plain_model.sequential_models[-1].init_student(self.plain_model[-1], torch.eye(widths[-2]))
            print('Tail initialized')


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
            freeze_BN(self.teacher)
        else:
            unfreeze_BN(self.teacher)

    def init_bridges(self):
        sample, _ = self.unpack_batch(self.train_dataloader().dataset[0])
        sample = sample.unsqueeze(dim=0)
        ret = nn.ModuleList()
        with torch.no_grad():
            feat_t, out_t = self.teacher(sample, with_feature=True)
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
                feat_t, out_t = self.teacher(images, with_feature=True)
            assert len(feat_s) == len(feat_t)
            dist_loss = self.dist_method(feat_s, feat_t, self.current_epoch / self.params['num_epochs'])

            mid_feature = self.forward(images, until=self.current_layer + 1)
            transfer_feature = self.bridges[self.current_layer](mid_feature)
            predictions = self.teacher_plain_model(transfer_feature, start_forward_from=self.current_layer + 1)
            task_loss = self.criterion(predictions, labels)
            loss = task_loss + dist_loss * self.params['distill_coe']

            self.log('train/dist_loss', dist_loss, sync_dist=True)
            self.log('train/task_loss', task_loss, sync_dist=True)
        else:
            mid_feature = self.forward(images, until=self.current_layer + 1)
            transfer_feature = self.bridges[self.current_layer](mid_feature)
            predictions = self.teacher_plain_model(transfer_feature, start_forward_from=self.current_layer + 1)
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
        self.init_layer(self.plain_model[0], self.teacher_plain_model.sequential_models[0], torch.eye(3))
        assert len(self.plain_model) == len(self.teacher_plain_model.sequential_models)

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
                                        self.teacher_plain_model.sequential_models[self.current_layer], M)
                    self.bridges[self.current_layer].weight.data = M.reshape(list(M.shape) + [1, 1]).cuda()
                else:
                    self.teacher_plain_model.sequential_models[-1].init_student(self.plain_model[-1], M)
                self.to(device)
            self.sync_plain_model()

    def step(self, batch, phase: str):
        images, labels = self.unpack_batch(batch)

        mid_feature = self.forward(images, until=self.current_layer + 1)
        transfer_feature = self.bridges[self.current_layer](mid_feature)
        predictions = self.teacher_plain_model(transfer_feature, start_forward_from=self.current_layer + 1)
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
        'DEIP_Dropout_Init': DEIP_Dropout_Init,
    }
    print("using method ", params['method'])
    model = methods[params['method']](params)
    return model
