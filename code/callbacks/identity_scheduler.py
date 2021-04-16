import torch
from pytorch_lightning.callbacks import Callback


class IDScheduler(Callback):
    def __init__(self, gamma):
        self.gamma = gamma
        self.current_step = 0
        self.last_val = 1

    def on_train_start(self, trainer, pl_module):
        """ Called before training, determines unique names for all lr
            schedulers in the case of multiple of the same type or in
            the case of multiple parameter groups
        """

        print('identity scheduler is used.')

    def _calc_val(self, step):
        # position = step / self.total_steps
        # return max(0, 1-position)
        return self.last_val

    def on_batch_start(self, trainer, pl_module):
        for name, module in pl_module.model.named_modules():
            if hasattr(module, "alpha_schedule"):
                with torch.no_grad():
                    module.alpha_schedule = self._calc_val(self.current_step)
        self.current_step = self.current_step + 1
        self.last_val = self.last_val * self.gamma
