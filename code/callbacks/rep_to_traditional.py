import collections
from pytorch_lightning.callbacks import Callback


class StructureChanger(Callback):
    def __init__(self, change_epoch=20):
        self.change_epoch = change_epoch

    def on_train_start(self, trainer, pl_module):
        """ Called before training, determines unique names for all lr
            schedulers in the case of multiple of the same type or in
            the case of multiple parameter groups
        """

        print('training structure changer is used.')

    def on_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        if trainer.current_epoch == self.change_epoch:
            for name, module in pl_module.model.named_modules():
                if hasattr(module, "to_traditional"):
                    module.to_traditional()
        for opt in trainer.optimizers:
            opt.state = collections.defaultdict(dict)
        pl_module.train()

