import sys
import os

sys.path.append(os.getcwd())

from frameworks.lightning_base_model import _Module
import utils.backend as backend
from pytorch_lightning.utilities import rank_zero_only


###
# python -m torch.distributed.launch --nproc_per_node=8 ../main.py
# nvidia-smi         发现内存泄露问题，即没有进程时，内存被占用
# kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
###

def get_params():
    from pytorch_lightning import seed_everything
    params = backend.load_parameters()
    seed_everything(params['seed'])
    backend.log_params(params)
    return params


def prepare_params(params):
    from utils.tools import parse_params
    params = parse_params(params)
    default_keys = {
        'inference_statics': True,
        'skip_train': False,
        'save_model': True,
        'metric': 'acc',
    }
    return {**default_keys, **params}


def train_model(model, params, save_name='default', checkpoint_monitor=None, mode='max'):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning import Trainer
    from utils.tools import get_trainer_params

    if checkpoint_monitor is None:
        checkpoint_monitor = 'validation/' + params['metric']

    logger = TensorBoardLogger("logs", name=save_name, default_hp_metric=False)
    backend.set_tensorboard_logdir(f'logs/{save_name}')

    checkpoint_callback = ModelCheckpoint(dirpath='saves', save_top_k=1, monitor=checkpoint_monitor, mode=mode)
    t_params = get_trainer_params(params)
    trainer = Trainer(logger=logger, callbacks=[checkpoint_callback], **t_params)
    trainer.fit(model)
    # trainer.test(model)
    model.eval()
    from utils.tools import get_model_weight_hash
    print(f'model weight hash {get_model_weight_hash(model)}')

    if checkpoint_callback.best_model_path != "" and trainer.is_global_zero:
        import numpy as np
        if params['save_model']:
            backend.save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
        log_val = checkpoint_callback.best_model_score.item()
        backend.log_metric(checkpoint_monitor.split('/')[-1], float(np.clip(log_val, -1e10, 1e10)))
    else:
        backend.log("Best_model_path not found!")

    backend.log("Training finished")
    return model


@rank_zero_only
def inference_statics(model, x_test=None, batch_size=None):
    import time
    import torch

    if x_test is None:
        x_test = model.val_dataloader().dataset[0][0]
    print('x_test size ', x_test.shape)
    if batch_size is None:
        batch_size = model.val_dataloader().batch_size
    x = torch.stack([x_test] * batch_size, dim=0).cuda()
    model.cuda().eval()
    total_input_size = x.nelement()
    with torch.no_grad():
        for i in range(10):
            outs = model(x)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        start_time = time.time()
        for i in range(100):
            outs = model(torch.randn_like(x))
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        used_memory = torch.cuda.max_memory_allocated()
        backend.log_metric('Inference_Time(us)', float(total_time / 100 / total_input_size * 1e6))  # time usage per pixel per batch
        backend.log_metric('Memory(KB)', float(used_memory / total_input_size / 1024))  # memory usage per pixel per batch

    from thop import profile
    x = torch.stack([x_test], dim=0).cuda()
    flops, param_number = profile(model, inputs=(x,), verbose=False)
    backend.log_metric('flops(K per pixel)', float(flops / x.nelement() / 1000))
    backend.log_metric('parameters(KB)', float(param_number / 1024))


if __name__ == "__main__":
    hparams = get_params()
    hparams = prepare_params(hparams)
    backend.log(hparams)
    pl_model = _Module(hparams).cuda()

    if not hparams['skip_train']:
        pl_model = train_model(pl_model, hparams, save_name='direct_train')

    if hparams['inference_statics']:
        inference_statics(pl_model, batch_size=256)
