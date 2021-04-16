import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class DataRecoder(Callback):
    def on_train_start(self, trainer, pl_module):
        """ Called before training, determines unique names for all lr
            schedulers in the case of multiple of the same type or in
            the case of multiple parameter groups
        """
        if not trainer.logger:
            raise MisconfigurationException(
                'Cannot use LearningRateLogger callback with Trainer that has no logger.')

        if not trainer.lr_schedulers:
            rank_zero_warn(
                'You are using LearningRateLogger callback with models that'
                ' have no learning rate schedulers. Please see documentation'
                ' for `configure_optimizers` method.', RuntimeWarning
            )

        print('data recorder used.')

    def on_epoch_end(self, trainer, pl_module):
        if trainer.logger and trainer.is_global_zero:
            print('logging record data')
            for name, module in pl_module.model.named_modules():
                if hasattr(module, "record_data"):
                    metrics = {key + '/' + name: torch.mean(torch.tensor(values)) for key, values in module.record_data.items()}
                    trainer.logger.log_metrics(metrics, step=trainer.global_step)
                    module.record_data = {key: [] for key in module.record_data.keys()}
