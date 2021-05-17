import sys
sys.path.append('/job/job_source/')

from frameworks.superresolution.SRModel import load_model
from datasets.dataProvider import DataProvider
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
    if params['backbone'] == 'VDSR':
        if isinstance(params['datasets'], dict):
            params['datasets']['input_large'] = True
        else:
            params['datasets'] = {'name': params['datasets'], 'input_large': True}
    default_keys = {
        'loss': 'L1',
        'self_ensemble': False,
        'inference_statics': False,
        'skip_train': False,
        'save_model': True,
        'test_benchmark': False,
    }
    params = {**default_keys, **params}
    return params


def train_model(model, params):
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import ModelCheckpoint

    logger = TensorBoardLogger("../logs", name='super_resolution')
    from foundations import set_tensorboard_logdir
    set_tensorboard_logdir(f'../logs/super_resolution')

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='validation/PSNR', mode='max',
                                          prefix=str(params["seed"]))
    t_params = get_trainer_params(params)
    trainer = Trainer(logger=logger, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=100,
                      **t_params)
    trainer.fit(model)

    if checkpoint_callback.best_model_path != "":
        from foundations import save_artifact

        if params['save_model']:
            save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
        log_val = checkpoint_callback.best_model_score.item()
        log_metric("val_PSNR", float(np.clip(log_val, -1e10, 1e10)))

    print("Training finished")
    return model


if __name__ == "__main__":
    params = get_params()
    params = prepare_params(params)
    print(params)

    model = load_model(params)

    if not params['skip_train']:
        model = train_model(model, params)

    if params['test_benchmark']:
        from meter.super_resolution_meter import SuperResolutionMeter

        benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']
        meter = SuperResolutionMeter(phase='test', workers=1, scale=2)
        for d in benchmarks:
            dataset_params = {
                'name': d,
                'test_only': True,
                'patch_size': params['dataset']['patch_size'],
                'ext': 'sep',
                'scale': params['scale'],
                "batch_size": 1,
            }
            provider = DataProvider(dataset_params)
            model.eval().cuda()
            with torch.no_grad():
                for batch in provider.test_dl:
                    x, y, _ = batch
                    model.step(meter, (x.cuda(), y.cuda(), _))
                metrics = meter.log_metric()
                meter.reset()
                log_metric(d + "_" + "PSNR_GRAY", float(np.clip(metrics['test/PSNR_GRAY'].item(), -1e10, 1e10)))
                # for k, v in metrics.items():
                #     log_metric(d + "_" + k, float(np.clip(v.item(), -1e10, 1e10)))

    if params['inference_statics']:
        import time

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        x_test = torch.randint(0, 256, (16, 3, 96, 96)).float().cuda()
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
