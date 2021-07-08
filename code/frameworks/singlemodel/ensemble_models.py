import torch

from frameworks.lightning_base_model import LightningModule
from model import get_classifier, load_models
from model.basic_cifar_models import ResNet
import copy
from torch import nn
import model.repdistiller_models as Rep_models


class Traditional_Ensemble(LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.models = load_models(hparams)
        self.num = len(self.models)
        self.modeles.cuda()

    def forward(self, images):
        predictions = torch.sum(torch.stack([model(images) for model in self.models]), dim=0)
        return predictions

    def configure_optimizers(self):
        return None, None
