# Distill Everything Into a Plane model

import torch
from torch import nn

from model import get_classifier, freeze
from frameworks.lightning_base_model import LightningModule
from model.basic_cifar_models.resnet_layerwise_cifar import LayerWiseModel, LastLinearLayer


# TODO: 首先实现一个确定宽度，然后直接 Train。
class DEIP_LightModel(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.teacher_model: LayerWiseModel = get_classifier(hparams["backbone"], hparams["dataset"])
        freeze(self.teacher_model.eval())

        self.plane_model = nn.ModuleList()
        self.teacher_start_layer = 0
        self.last_channel = self.hparams['input_channel']
        self.init_student()

    def init_student(self):
        import time
        start_time = time.clock()
        # First version, no progressive learning
        for batch in self.dataProvider.train_dl[:1]:
            widths = self.calc_width(input_batch=batch)
            print("calculated width = ", widths)
            for w in widths[:-1]:
                self.append_layer(w)
            self.append_fc(widths[-1])
            print("initialization student width used ", time.clock() - start_time)
            break

    def complete_hparams(self):
        default_sr_list = {
            'input_channel': 3,
            'progressive_distillation': False,
        }
        self.hparams = {**default_sr_list, **self.hparams}
        LightningModule.complete_hparams(self)

    def forward(self, x, with_feature=False, start_forward_from=0, until=None):
        if not with_feature:
            for m in self.plane_model:
                x = m(x)
            return x
        else:
            f_list = []
            for m in self.plane_model[start_forward_from: until]:
                x = m(x)
                f_list.append(x)
            return f_list, x

    def step(self, meter, batch):
        images, labels = batch
        predictions = self.forward(images)
        loss = self.criterion(predictions, labels)
        meter.update(labels, predictions.detach(), loss.detach())
        acc = (torch.max(predictions, dim=1)[1] == labels).float().mean()
        return {'loss': loss, 'progress_bar': {'acc': acc}}

    def append_layer(self, channels, kernel_size=3):
        new_layer = nn.Conv2d(self.last_channel, channels, kernel_size=kernel_size)
        self.last_channel = channels
        self.plane_model.append(new_layer)

    def append_fc(self, num_classes):
        self.plane_model.append(LastLinearLayer(self.last_channel, num_classes))

    def calc_width(self, input_batch):
        if self.hparams['progressive_distillation']:  # progressive 更好会不会是训得更久所以效果更好
            # TODO: calculate next layer width
            pass
        else:
            ret = []
            with torch.no_grad():
                images, labels = input_batch
                f_list, _ = self.teacher_model(images, with_feature=True)
                for f in f_list[:-1]:
                    #  Here Simple SVD is used, which is the best approximation to min_{D'} ||D-D'||_F where rank(D') <= r
                    #  A question is, how to solve min_{D'} ||(D-D')*W||_F where rank(D') <= r, W is matrix with positive weights and * is element-wise production
                    #  refer to wiki, it's called `Weighted low-rank approximation problems`, which does not have an analytic solution
                    ret.append(rank_estimate(f))
                ret.append(f_list[-1].size(1))  # num classes
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
    u, s, v = torch.svd(f)
    error = 0
    for r in range(1, f.size(0) + 1):
        approx = torch.mm(torch.mm(u[:, :r], torch.diag(s[:r])), v[:, :r].t())
        error = torch.max(torch.abs(f - approx) / (torch.abs(f)+1e-4))
        if error < eps:
            return r
    raise AssertionError(f"rank estimation failed! The last error is {error}")

# TODO: DEIP_Distillation, DEIP_Progressive_Distillation


def load_model(params):
    methods = {
        'DirectTrain': DEIP_LightModel,
    }
    if 'method' not in params:
        Selected_Model = DEIP_LightModel
    else:
        Selected_Model = methods[params['method']]

    model = Selected_Model(params)
    return model
