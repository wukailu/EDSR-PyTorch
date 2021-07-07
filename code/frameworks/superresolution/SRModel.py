import torch

from frameworks.distillation.feature_distillation import get_distill_module
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
        self.params = {**default_sr_list, **self.params}
        LightningModule.complete_hparams(self)

    def choose_loss(self):
        from .loss import Loss
        return Loss(self.params)

    def get_meter(self, phase: str):
        from meter.super_resolution_meter import SuperResolutionMeter as Meter
        workers = 1 if phase == "test" or self.params["backend"] != "ddp" else self.params["gpus"]
        return Meter(phase=phase, workers=workers, scale=self.params['scale'])

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
        self.dist_method = self.get_distillation_module()

    def load_teacher(self):
        model = load_model({'load_from': self.params['teacher']}).model
        from model.utils import freeze
        freeze(model)
        return model

    def get_distillation_module(self):
        sample, _, _ = self.train_dataloader().dataset[0]  # TODO: get this message from data provider
        sample = sample.unsqueeze(dim=0)
        with torch.no_grad():
            feat_t, out_t = self.teacher(sample, with_feature=True)
            feat_s, out_s = self.model(sample, with_feature=True)
            dist_method = get_distill_module(self.params['dist_method'])(feat_s, feat_t)
        return dist_method

    def complete_hparams(self):
        default_sr_list = {
            'distill_coe': 1,
            'start_distill': 0,
            'pretrain_distill': False,
            'dist_method': 'L2Distillation',
        }
        self.params = {**default_sr_list, **self.params}
        SR_LightModel.complete_hparams(self)

    def step(self, meter, batch):
        lr, hr, filenames = batch
        if self.training:
            feat_s, out_s = self.model(lr, with_feature=True)
            task_loss = self.criterion(out_s, hr)
            if self.current_epoch < self.params['start_distill'] and not self.params['pretrain_distill']:
                loss = task_loss
            else:
                feat_t, out_t = self.teacher(lr, with_feature=True)
                if self.current_epoch < self.params['start_distill'] and self.params['pretrain_distill']:
                    dist_loss = self.dist_method([fs.detach() for fs in feat_s], [ft.detach() for ft in feat_t], self.current_epoch/self.params['num_epochs'])
                else:
                    dist_loss = self.dist_method(feat_s, feat_t, self.current_epoch/self.params['num_epochs'])
                self.logger.log_metrics({'train/dist_loss': dist_loss.detach()}, step=self.global_step)
                loss = task_loss + dist_loss * self.params['distill_coe']

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
        self.params = {**default_list, **self.params}
        SRDistillation.complete_hparams(self)

    def step(self, meter, batch):
        teacher_dict = self.teacher.state_dict()
        student_dict = self.model.state_dict()
        alpha = self.params['mean_teacher_momentum']
        for key, value in student_dict.items():
            teacher_dict[key] = teacher_dict[key] * alpha + value * (1 - alpha)
        self.teacher.load_state_dict(teacher_dict)

        return SRDistillation.step(self, meter, batch)


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
