import torch

from model import get_classifier
from frameworks.lightning_base_model import LightningModule


class SR_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.scale = hparams['scale']
        self.idx_scale = 0
        self.self_ensemble = hparams['self_ensemble']
        self.model = get_classifier(hparams["backbone"], hparams["dataset"])

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
        # self.idx_scale = idx_scale
        # if hasattr(self.model, 'set_scale'):
        #     self.model.set_scale(idx_scale)

        if self.training:
            return self.do_forward(x)
        else:
            # if self.chop:
            #     forward_function = self.forward_chop
            # else:
            #     forward_function = self.model.forward

            forward_function = self.do_forward
            if self.self_ensemble:
                return self.forward_x8(x, forward_function=forward_function)
            else:
                return forward_function(x)

    # def forward_chop(self, *args, shave=10, min_size=160000):
    #     scale = 1 if self.input_large else self.scale[self.idx_scale]
    #     n_GPUs = min(self.n_GPUs, 4)
    #     # height, width
    #     h, w = args[0].size()[-2:]
    #
    #     top = slice(0, h//2 + shave)
    #     bottom = slice(h - h//2 - shave, h)
    #     left = slice(0, w//2 + shave)
    #     right = slice(w - w//2 - shave, w)
    #     x_chops = [torch.cat([
    #         a[..., top, left],
    #         a[..., top, right],
    #         a[..., bottom, left],
    #         a[..., bottom, right]
    #     ]) for a in args]
    #
    #     y_chops = []
    #     if h * w < 4 * min_size:
    #         for i in range(0, 4, n_GPUs):
    #             x = [x_chop[i:(i + n_GPUs)] for x_chop in x_chops]
    #             y = P.data_parallel(self.model, *x, range(n_GPUs))
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[c for c in _y.chunk(n_GPUs, dim=0)] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y):
    #                     y_chop.extend(_y.chunk(n_GPUs, dim=0))
    #     else:
    #         for p in zip(*x_chops):
    #             y = self.forward_chop(*p, shave=shave, min_size=min_size)
    #             if not isinstance(y, list): y = [y]
    #             if not y_chops:
    #                 y_chops = [[_y] for _y in y]
    #             else:
    #                 for y_chop, _y in zip(y_chops, y): y_chop.append(_y)
    #
    #     h *= scale
    #     w *= scale
    #     top = slice(0, h//2)
    #     bottom = slice(h - h//2, h)
    #     bottom_r = slice(h//2 - h, None)
    #     left = slice(0, w//2)
    #     right = slice(w - w//2, w)
    #     right_r = slice(w//2 - w, None)
    #
    #     # batch size, number of color channels
    #     b, c = y_chops[0][0].size()[:-2]
    #     y = [y_chop[0].new(b, c, h, w) for y_chop in y_chops]
    #     for y_chop, _y in zip(y_chops, y):
    #         _y[..., top, left] = y_chop[0][..., top, left]
    #         _y[..., top, right] = y_chop[1][..., top, right_r]
    #         _y[..., bottom, left] = y_chop[2][..., bottom_r, left]
    #         _y[..., bottom, right] = y_chop[3][..., bottom_r, right_r]
    #
    #     if len(y) == 1: y = y[0]
    #
    #     return y

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
        if len(y) == 1: y = y[0]

        return y


class TwoStageSR(SR_LightModel):
    def __init__(self, hparams):
        LightningModule.__init__(self, hparams)
        self.scale = hparams['scale']
        self.idx_scale = 0
        self.self_ensemble = hparams['self_ensemble']

        self.model_pretrained = SR_LightModel.load_from_checkpoint(checkpoint_path=hparams['pretrained_from']).model
        self.model = SR_LightModel.load_from_checkpoint(checkpoint_path=hparams['pretrained_from']).model
        if 'two_stage_no_freeze' not in hparams or not hparams['two_stage_no_freeze']:
            from model.utils import freeze
            freeze(self.model_pretrained)

    # x2 + x2 = x4
    def do_forward(self, x):
        return self.model_pretrained(self.model(x))


def load_model(params):
    methods = {
        'TwoStageSR': TwoStageSR,
        'SR_LightModel': SR_LightModel,
    }
    if 'method' not in params:
        LightModel = SR_LightModel
    else:
        LightModel = methods[params['method']]

    if 'load_from' in params:
        path = params['load_from']
        assert isinstance(path, str)
        model = LightModel.load_from_checkpoint(checkpoint_path=path).cuda()
    elif 'load_model_from' in params:
        path = params['load_model_from']
        assert isinstance(path, str)
        model_inside = SR_LightModel.load_from_checkpoint(checkpoint_path=path).model.cuda()
        model = LightModel(params).cuda()
        model.model = model_inside
    else:
        model = LightModel(params).cuda()
    return model