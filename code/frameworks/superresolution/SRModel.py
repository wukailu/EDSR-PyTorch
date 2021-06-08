import torch

from model import get_classifier
from frameworks.lightning_base_model import LightningModule


class SR_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.scale = hparams['scale']
        self.self_ensemble = hparams['self_ensemble']
        self.model = get_classifier(hparams["backbone"], hparams["dataset"])

    def complete_hparams(self):
        default_sr_list = {
            'loss': 'L1',
            'self_ensemble': False,
        }
        self.hparams = {**default_sr_list, **self.hparams}
        LightningModule.complete_hparams(self)

    def choose_loss(self):
        from .loss import Loss
        return Loss(self.hparams)

    def get_meter(self, phase: str):
        from meter.super_resolution_meter import SuperResolutionMeter as Meter
        workers = 1 if phase == "test" or self.hparams["backend"] != "ddp" else self.hparams["gpus"]
        return Meter(phase=phase, workers=workers, scale=self.hparams['scale'])

    def step(self, meter, batch):
        lr, hr, filenames = batch
        predictions = self.forward(lr)
        loss = self.criterion(predictions, hr)
        meter.update(hr, predictions.detach(), loss.detach())
        return {'loss': loss}

    def do_forward(self, x):
        return self.model(x)

    def forward(self, x):
        if self.training:
            return self.do_forward(x)
        else:
            forward_function = self.do_forward
            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    def forward_x8(self, *args, forward_function=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = forward_function(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1:
            y = y[0]
        return y


class TwoStageSR(SR_LightModel):
    def __init__(self, hparams):
        LightningModule.__init__(self, hparams)
        self.scale = hparams['scale']
        self.self_ensemble = hparams['self_ensemble']

        self.model_pretrained = SR_LightModel.load_from_checkpoint(checkpoint_path=hparams['pretrained_from']).model
        self.model = SR_LightModel.load_from_checkpoint(checkpoint_path=hparams['pretrained_from']).model
        if 'two_stage_no_freeze' not in hparams or not hparams['two_stage_no_freeze']:
            from model.utils import freeze
            freeze(self.model_pretrained)

    # x2 + x2 = x4
    def do_forward(self, x):
        return self.model_pretrained(self.model(x))


class SRDistillation(SR_LightModel):
    def __init__(self, hparams):
        LightningModule.__init__(self, hparams)
        self.scale = hparams['scale']
        self.self_ensemble = hparams['self_ensemble']
        self.model = get_classifier(hparams["backbone"], hparams["dataset"])

        self.teacher = self.load_teacher()
        self.dist_method = self.init_distill()

    def load_teacher(self):
        model = load_model({'load_from': self.hparams['teacher']}).model
        from model.utils import freeze
        freeze(model)
        return model

    def init_distill(self):
        sample, _, _ = self.train_dataloader().dataset[0]  # TODO: get this message from data provider
        sample = sample.unsqueeze(dim=0)
        with torch.no_grad():
            out_t, feat_t = self.teacher(sample, with_feature=True)
            out_s, feat_s = self.model(sample, with_feature=True)
            dist_method = get_distill_method(self.hparams['dist_method'])(feat_s, feat_t)
        return dist_method

    def complete_hparams(self):
        default_sr_list = {
            'distill_coe': 1,
            'start_distill': 0,
            'pretrain_distill': False,
            'dist_method': 'L2Distillation',
        }
        self.hparams = {**default_sr_list, **self.hparams}
        SR_LightModel.complete_hparams(self)

    def step(self, meter, batch):
        lr, hr, filenames = batch
        if self.training:
            out_s, feat_s = self.model(lr, with_feature=True)
            task_loss = self.criterion(out_s, hr)
            if self.current_epoch < self.hparams['start_distill'] and not self.hparams['pretrain_distill']:
                loss = task_loss
            else:
                out_t, feat_t = self.teacher(lr, with_feature=True)
                if self.current_epoch < self.hparams['start_distill'] and self.hparams['pretrain_distill']:
                    dist_loss = self.dist_method([fs.detach() for fs in feat_s], [ft.detach() for ft in feat_t])
                else:
                    dist_loss = self.dist_method(feat_s, feat_t)
                self.logger.log_metrics({'train/dist_loss': dist_loss.detach()}, step=self.global_step)
                loss = task_loss + dist_loss * self.hparams['distill_coe']

            self.logger.log_metrics({'train/task_loss': task_loss.detach()}, step=self.global_step)
        else:
            out_s = self.forward(lr)
            loss = self.criterion(out_s, hr)

        meter.update(hr, out_s.detach(), loss.detach())
        return {'loss': loss}


class MeanTeacherSRDistillation(SRDistillation):
    def load_teacher(self):
        import copy
        from model.utils import freeze
        model = copy.deepcopy(self.model)
        freeze(model)
        return model

    def complete_hparams(self):
        default_list = {
            'mean_teacher_momentum': 0.9,
        }
        self.hparams = {**default_list, **self.hparams}
        SRDistillation.complete_hparams(self)

    def step(self, meter, batch):
        teacher_dict = self.teacher.state_dict()
        student_dict = self.model.state_dict()
        alpha = self.hparams['mean_teacher_momentum']
        for key, value in student_dict.items():
            teacher_dict[key] = teacher_dict[key] * alpha + value * (1 - alpha)
        self.teacher.load_state_dict(teacher_dict)

        return SRDistillation.step(self, meter, batch)


def get_distill_method(name):
    methods = {
        'CKA': CKA,
        'L2Distillation': L2Distillation,
        'L1Distillation': L1Distillation,
        'FD_Conv1x1': FD_Conv1x1,
        'FD_CloseForm': FD_CloseForm,
        'FD_BN1x1': FD_BN1x1,
        'FD_Conv1x1_MSE': FD_Conv1x1_MSE
    }
    return methods[name]


class DistillationMethod(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t):
        pass


class L2Distillation(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t):
        loss = []
        for fs, ft in zip(feat_s, feat_t):
            loss.append(torch.mean((fs - ft)**2))
        return torch.mean(torch.stack(loss))


class L1Distillation(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t):
        loss = []
        for fs, ft in zip(feat_s, feat_t):
            loss.append(torch.mean(torch.abs(fs - ft)))
        return torch.mean(torch.stack(loss))


class FD_Conv1x1(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t)
        ])

    def forward(self, feat_s, feat_t):
        loss = 0
        for fs, ft, conv in zip(feat_s, feat_t, self.convs):
            loss += torch.mean(torch.abs(conv(fs) - ft))
        return loss


class FD_Conv1x1_MSE(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t)
        ])

    def forward(self, feat_s, feat_t):
        loss = []
        for fs, ft, conv in zip(feat_s, feat_t, self.convs):
            loss.append(torch.mean((conv(fs) - ft)**2))
        return torch.mean(torch.stack(loss))


class FD_BN1x1(DistillationMethod):
    def __init__(self, feat_s, feat_t, *args, **kwargs):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv2d(fs.size(1), ft.size(1), kernel_size=1) for fs, ft in zip(feat_s, feat_t)
        ])
        self.bn_t = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(ft.size(1)) for fs, ft in zip(feat_s, feat_t)
        ])
        self.bn_s = torch.nn.ModuleList([
            torch.nn.BatchNorm2d(fs.size(1)) for fs, ft in zip(feat_s, feat_t)
        ])

    def forward(self, feat_s, feat_t):
        loss = 0
        for fs, ft, conv, bn_t, bn_s in zip(feat_s, feat_t, self.convs, self.bn_t, self.bn_s):
            ft = bn_t(ft)
            fs = bn_s(fs)
            loss += torch.mean(torch.abs(conv(fs) - ft))
        return loss


class FD_CloseForm(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, feat_s, feat_t):
        loss = 0
        for fs, ft in zip(feat_s, feat_t):
            # 1x1 conv equivalent to AX = B, where A is of shape [N, HxW, C_s], X is [N, C_s, C_t], B is [N, HxW, C_t]
            # lstsq with batch is available in torch.linalg.lstsq with torch >= 1.9.0
            # torch.lstsq does not support grad
            A = torch.flatten(fs, start_dim=2).permute((0, 2, 1))
            B = torch.flatten(ft, start_dim=2).permute((0, 2, 1))

            # MSE over batches, this might be numerical unstable
            f_cnt = 0
            flag = False
            while not flag:
                flag = True
                A += torch.randn_like(A) * 0.1
                try:
                    s = A.pinverse() @ B
                except RuntimeError as e:
                    flag = False
                    f_cnt += 1
                if not flag:
                    A += torch.randn_like(A) * 0.1
            if f_cnt != 0:
                print(f"pinverse failed! repeated {f_cnt} times to get succeeded.")
                assert False

            r = (A @ s - B)
            loss += torch.mean(r ** 2)
        return loss


class CKA(DistillationMethod):
    def __init__(self, *args, **kwargs):
        super().__init__()
        from frameworks.nnmetric.feature_similarity_measurement import cka_loss
        self.cka = cka_loss()

    def forward(self, feat_s, feat_t):
        loss = 0
        for fs, ft in zip(feat_s, feat_t):
            loss += self.cka(fs, ft)
        return loss


def load_model(params):
    methods = {
        'TwoStageSR': TwoStageSR,
        'SR_LightModel': SR_LightModel,
        'SRDistillation': SRDistillation,
        'MeanTeacherSRDistillation': MeanTeacherSRDistillation,
    }
    if 'method' not in params:
        Selected_Model = SR_LightModel
    else:
        Selected_Model = methods[params['method']]

    if 'load_from' in params:
        path = params['load_from']
        assert isinstance(path, str)
        try:
            model = Selected_Model.load_from_checkpoint(checkpoint_path=path)
        except TypeError as e:
            cp = torch.load(path)
            model = Selected_Model(cp['hyper_parameters'])
            model.load_state_dict(cp['state_dict'])
    elif 'load_model_from' in params:
        path = params['load_model_from']
        assert isinstance(path, str)
        model_inside = SR_LightModel.load_from_checkpoint(checkpoint_path=path).model
        model = Selected_Model(params)
        model.model = model_inside
    else:
        model = Selected_Model(params)
    return model
