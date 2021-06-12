import sys
sys.path.append('/job/job_source/')

from frameworks.distillation.DEIP import load_model
import torch
from pytorch_lightning import Trainer, seed_everything
from utils.foundation_tools import get_trainer_params
from foundations import log_metric
import numpy as np


###
# python -m torch.distributed.launch --nproc_per_node=8 ../main.py
# nvidia-smi         发现内存泄露问题，即没有进程时，内存被占用
# kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
###

def get_params():
    from foundations import load_parameters, log_params
    print("using atlas framework")
    params = load_parameters()
    seed_everything(params['seed'])
    log_params(params)
    return params


def prepare_params(params):
    from utils.foundation_tools import parse_params
    params = parse_params(params)
    default_keys = {
        'inference_statics': True,
        'skip_train': False,
        'save_model': True,
    }
    params = {**default_keys, **params}
    return params


def direct_train_model(model, params):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    logger = TensorBoardLogger("../logs", name='model_distillation')
    from foundations import set_tensorboard_logdir
    set_tensorboard_logdir(f'../logs/model_distillation')

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='validation/acc', prefix=str(params["seed"]))
    t_params = get_trainer_params(params)
    trainer = Trainer(logger=logger, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=100, **t_params)
    trainer.fit(model)
    trainer.test(model)

    if checkpoint_callback.best_model_path != "":
        from foundations import save_artifact

        if params['save_model']:
            save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
        log_val = checkpoint_callback.best_model_score.item()
        log_metric("acc", float(np.clip(log_val, -1e10, 1e10)))

    print("Training finished")
    return model


if __name__ == "__main__":
    params = get_params()
    params = prepare_params(params)
    print(params)

    model = load_model(params)

    if not params['skip_train']:
        model = direct_train_model(model, params)

    if params['inference_statics']:
        import time

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        x_test = torch.stack([model.val_dataloader().dataset[0][0]]*16, dim=0).float().cuda()
        model.cuda().eval()
        with torch.no_grad():
            total_time = 0
            for i in range(10):
                outs = model(x_test)
            for i in range(20):
                torch.cuda.synchronize()
                start_time = time.process_time()
                outs = model(x_test)
                total_time += time.process_time() - start_time
            used_memory = torch.cuda.max_memory_allocated()
            log_metric('Inference_Time(ms)', float(total_time / 20 * 1000))
            log_metric('Memory(MB)', int(used_memory/1024/1024))

        from thop import profile
        x_test = torch.stack([model.val_dataloader().dataset[0][0]], dim=0).float().cuda()
        flops, params = profile(model, inputs=(x_test,))
        log_metric('flops(M)', float(flops/1024/1024))
        log_metric('parameters(K)', float(params / 1024))
