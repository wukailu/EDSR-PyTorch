import sys
import os

sys.path.append(os.getcwd())

from frameworks.lightning_base_model import _Module
import utils.backend as backend


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

    if checkpoint_callback.best_model_path != "":
        import numpy as np
        if params['save_model']:
            backend.save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
        log_val = checkpoint_callback.best_model_score.item()
        backend.log_metric(checkpoint_monitor.split('/')[-1], float(np.clip(log_val, -1e10, 1e10)))
    else:
        backend.log("Best_model_path not found!")

    backend.log("Training finished")
    return model


def inference_statics(model, x_test=None, batch_size=16):
    import time
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    if x_test is None:
        x_test = model.val_dataloader().dataset[0][0]
    x = torch.stack([x_test] * batch_size, dim=0).cuda()
    model.cuda().eval()
    with torch.no_grad():
        total_time = 0
        for i in range(10):
            outs = model(x)
        for i in range(20):
            torch.cuda.synchronize()
            start_time = time.process_time()
            outs = model(x)
            total_time += time.process_time() - start_time
        used_memory = torch.cuda.max_memory_allocated()
        backend.log_metric('Inference_Time(ms)', float(total_time / 20 * 1000))
        backend.log_metric('Memory(MB)', int(used_memory / 1024 / 1024))

    from thop import profile
    x = torch.stack([x_test], dim=0).cuda()
    flops, param_number = profile(model, inputs=(x,))
    backend.log_metric('flops(M)', float(flops / 1024 / 1024))
    backend.log_metric('parameters(K)', float(param_number / 1024))


if __name__ == "__main__":
    hparams = get_params()
    hparams = prepare_params(hparams)
    backend.log(hparams)
    pl_model = _Module(hparams).cuda()

    if not hparams['skip_train']:
        pl_model = train_model(pl_model, hparams, save_name='direct_train')

    if hparams['inference_statics']:
        inference_statics(pl_model)
