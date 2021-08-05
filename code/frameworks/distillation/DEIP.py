# Distill Everything Into a Plain model

import torch
from torch import nn

from model import get_classifier, freeze, unfreeze_BN, ConvertibleLayer
from frameworks.lightning_base_model import LightningModule, _Module
from model.basic_cifar_models.resnet_layerwise_cifar import ResNet_CIFAR, LastLinearLayer
from frameworks.distillation.feature_distillation import get_distill_module


class DEIP_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.M_maps = []
        self.teacher_model = self.load_teacher()
        # self.teacher_model = get_classifier('resnet20_layerwise', 'cifar100')
        freeze(self.teacher_model.eval())

        self.plane_model = nn.ModuleList()
        self.init_student()

    def load_teacher(self):
        return _Module.load_from_checkpoint(checkpoint_path=self.params['teacher_pretrain_path']).model

    def init_student(self):
        import time
        start_time = time.process_time()
        # First version, no progressive learning
        for batch in self.dataProvider.train_dl:
            widths = [self.params['input_channel']] + self.calc_width(batch=batch)
            if self.params['task'] == 'super-resolution':
                for i in range(1, len(widths)):  # TODO: fix this bug in a better way
                    widths[i] = min(widths[i - 1] * 9, widths[i])

            with torch.no_grad():
                images, _ = self.decompress_batch(batch)
                f_list, _ = self.teacher_model(images, with_feature=True)
                f_shapes = [images.shape] + [f.shape for f in f_list]
                for idx in range(len(widths) - 2):
                    self.append_layer(widths[idx], widths[idx + 1], f_shapes[idx][2:], f_shapes[idx + 1][2:])
            self.append_fc(widths[-2], widths[-1])
            print("initialization student width used ", time.process_time() - start_time)
            break

        if self.params['init_stu_with_teacher']:
            print("init student with teacher!")
            # TODO: Implement initialize student with teacher of Method 2
            # teacher and student has different #input_channel and #out_channel
            # Method 1: ignore the relu layer, then student can be initialized at start
            # Method 1, branch 2: merge bias into conv by adding a constant channel to feature map
            # Method 2: Feature map mapping is kept by layer distillation, use the matrix from distillation, but this
            # need to be calculated on-the-fly.
            # Choice 1: Shall we restrict the feature map before relu or after relu?
            # Note 1: Pay attention to gradient vanishing after intialization.

            assert len(self.plane_model) == len(self.teacher_model.sequential_models)
            # Implement Method 1:
            M = torch.eye(3)  # M is of shape C_t x C_s
            for layer_s, layer_t in zip(self.plane_model[:-1], self.teacher_model.sequential_models[:-1]):
                assert isinstance(layer_t, ConvertibleLayer)
                if 'normal' in self.params['layer_type']:
                    M = layer_t.init_student(layer_s[0], M)
                elif 'plain_sr' in self.params['layer_type']:
                    M = layer_t.init_student(layer_s.conv, M)
                    if layer_s.skip:
                        k = layer_s.conv.kernel_size[0]
                        for i in range(layer_s.conv.out_channels):
                            layer_s.conv.weight.data[i, i, k // 2, k // 2] -= 1
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
                else:
                    raise NotImplementedError()
                self.M_maps.append(M.detach())

            # LastFCLayer
            self.teacher_model.sequential_models[-1].init_student(self.plane_model[-1], M)

    def complete_hparams(self):
        default_sr_list = {
            'task': 'classification',
            'input_channel': 3,
            'progressive_distillation': False,
            'rank_eps': 5e-2,
            'use_bn': True,
            'layer_type': 'normal',
            'init_stu_with_teacher': False,
            'teacher_pretrain_path': None,
        }
        self.params = {**default_sr_list, **self.params}
        LightningModule.complete_hparams(self)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        f_list = []
        for m in self.plane_model[start_forward_from: until]:
            x = m(x)
            f_list.append(x)
        return (f_list, x) if with_feature else x

    def decompress_batch(self, batch):
        if self.params['task'] == 'classification':
            images, labels = batch
        elif self.params['task'] == 'super-resolution':
            images, labels, filenames = batch
        else:
            raise NotImplementedError()
        return images, labels

    def step(self, batch, phase: str):
        images, labels = self.decompress_batch(batch)
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric)
        return loss

    def append_layer(self, in_channels, out_channels, previous_f_size, current_f_size, kernel_size=3):
        if self.params['layer_type'].startswith('normal'):
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
            if 'no_bn' not in self.params['layer_type']:
                new_layers.append(nn.BatchNorm2d(out_channels))
            if 'prelu' in self.params['layer_type']:
                new_layers.append(nn.PReLU())
            else:
                new_layers.append(nn.ReLU())
            new_layer = nn.Sequential(*new_layers)
        elif self.params['layer_type'] == 'repvgg':
            from model.basic_cifar_models.repvgg import RepVGGBlock
            stride_w = previous_f_size[0] // current_f_size[0]
            stride_h = previous_f_size[1] // current_f_size[1]
            new_layer = RepVGGBlock(in_channels, out_channels, kernel_size, stride=(stride_w, stride_h),
                                    padding=kernel_size // 2)
        elif self.params['layer_type'].startswith('plain_sr'):
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

        self.plane_model.append(new_layer)

    def append_fc(self, last_channel, output_channel):
        if self.params['task'] == 'classification':
            self.plane_model.append(LastLinearLayer(last_channel, output_channel))
        elif self.params['task'] == 'super-resolution':
            from model.super_resolution_model.edsr_layerwise_model import EDSRTail, EDSR_layerwise_Model
            if isinstance(self.teacher_model, EDSR_layerwise_Model):
                self.plane_model.append(
                    EDSRTail(self.params['scale'], last_channel, output_channel, 3, self.params['rgb_range']))
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def calc_width(self, batch):
        if self.params['progressive_distillation']:  # progressive 更好会不会是训得更久所以效果更好
            # TODO: calculate next layer width
            pass
        else:
            ret = []
            with torch.no_grad():
                images, labels = self.decompress_batch(batch)
                f_list, _ = self.teacher_model(images, with_feature=True)
                teacher = [f.size(1) for f in f_list]
                for f in f_list[:-2]:
                    #  Here Simple SVD is used, which is the best approximation to min_{D'} ||D-D'||_F where rank(D') <= r
                    #  A question is, how to solve min_{D'} ||(D-D')*W||_F where rank(D') <= r, W is matrix with positive weights and * is element-wise production
                    #  refer to wiki, it's called `Weighted low-rank approximation problems`, which does not have an analytic solution
                    ret.append(rank_estimate(f, eps=self.params['rank_eps']))
                ret.append(f_list[-2].size(1))
                ret.append(f_list[-1].size(1))  # num classes
            print("calculated teacher width = ", teacher)
            print("calculated student width = ", ret)
            return ret


def rank_estimate(feature, weight=None, eps=5e-2):
    """
    Estimate the size of feature map to approximate this.
    :param feature: tensor of shape (N, C, *)
    :param weight: tensor of shape (1, C, *) where is the importance weight on each neural
    :param eps: the error bar for low_rank approximation
    """
    f = feature.flatten(start_dim=2)
    if weight is not None:
        f = f * weight
    f = f.permute((1, 2, 0)).flatten(start_dim=1)

    # svd can not solve too large feature maps, so take samples
    if f.size(1) > 32768:
        perm = torch.randperm(f.size(1))[:32768]
        f = f[:, perm]
    u, s, v = torch.svd(f)

    error = 0
    for r in range(1, f.size(0) + 1):
        # print('now at ', r, 'error = ', error)
        approx = torch.mm(torch.mm(u[:, :r], torch.diag(s[:r])), v[:, :r].t())
        error = torch.max(torch.abs(f - approx))  # take absolute error, you can use weight to balance it.
        if error < eps:
            return r
    print("rank estimation failed! feature shape is ", feature.shape)
    print("max value and min value in feature is ", feature.max(), feature.min())
    raise AssertionError(f"rank estimation failed! The last error is {error}")


# TODO: 啥时候把蒸馏改成多继承

class DEIP_Distillation(DEIP_LightModel):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.dist_method = self.get_distillation_module()

    def get_distillation_module(self):
        sample, _ = self.train_dataloader().dataset[0]
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
        images, labels = self.decompress_batch(batch)

        if self.training:
            feat_s, predictions = self(images, with_feature=True)
            task_loss = self.criterion(predictions, labels)

            with torch.no_grad():
                feat_t, out_t = self.teacher_model(images, with_feature=True)
            assert len(feat_s) == len(feat_t)
            dist_loss = self.dist_method(feat_s, feat_t, self.current_epoch / self.params['num_epochs'])
            loss = task_loss + dist_loss * self.params['distill_coe']

            self.log('train/dist_loss', dist_loss)
            self.log('train/task_loss', task_loss)
        else:
            predictions = self.forward(images)
            loss = self.criterion(predictions, labels)
            teacher_pred = self.teacher_model(images)
            metric = self.metric(teacher_pred, labels)
            self.log(phase + '/' + 'teacher_' + self.params['metric'], metric)

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric)
        return loss


class DEIP_Progressive_Distillation(DEIP_Distillation):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.current_layer = 0
        self.milestone_epochs = list(
            range(0, self.params['num_epochs'], self.params['num_epochs'] // len(self.plane_model)))[1:]
        print(f'there are totally {len(self.plane_model)} layers in student, milestones are {self.milestone_epochs}')
        self.bridges = self.init_bridges()
        unfreeze_BN(self.teacher_model)

    def init_bridges(self):
        sample, _ = self.train_dataloader().dataset[0]
        sample = sample.unsqueeze(dim=0)
        ret = nn.ModuleList()
        with torch.no_grad():
            feat_t, out_t = self.teacher_model(sample, with_feature=True)
            feat_s, out_s = self(sample, with_feature=True)
            for fs, ft in zip(feat_s, feat_t):
                if fs.shape != ft.shape:
                    ret.append(nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1))
                else:
                    ret.append(nn.Identity())
        return ret

    def step(self, batch, phase: str):
        if self.current_epoch in self.milestone_epochs:
            print(f'freezing layer {self.current_layer}')
            freeze(self.plane_model[self.current_layer])
            self.current_layer += 1
            self.milestone_epochs = self.milestone_epochs[1:]

        images, labels = self.decompress_batch(batch)

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

            self.log('train/dist_loss', dist_loss)
            self.log('train/task_loss', task_loss)
        else:
            mid_feature = self.forward(images, until=self.current_layer + 1)
            transfer_feature = self.bridges[self.current_layer](mid_feature)
            predictions = self.teacher_model(transfer_feature, start_forward_from=self.current_layer + 1)
            loss = self.criterion(predictions, labels)

        metric = self.metric(predictions, labels)
        self.log(phase + '/' + self.params['metric'], metric)
        return loss


def load_model(params):
    params = {'method': 'DirectTrain', **params}
    methods = {
        'DirectTrain': DEIP_LightModel,
        'Distillation': DEIP_Distillation,
        'Progressive_Distillation': DEIP_Progressive_Distillation,
    }
    model = methods[params['method']](params)
    return model
