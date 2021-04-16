import sys
sys.path.append('/job/job_source/')

from frameworks.SuperResolution.SRModel import SR_LightModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.foundation_tools import parse_params, get_trainer_params

###
# python -m torch.distributed.launch --nproc_per_node=8 ../main.py
# nvidia-smi         发现内存泄露问题，即没有进程时，内存被占用
# kill $(ps aux | grep "main.py" | grep -v grep | awk '{print $2}')
###

from foundations import load_parameters, log_params
print("using atlas framework")
params = load_parameters()
seed_everything(params['seed'])
log_params(params)

params = parse_params(params)
print(params)

if params['backbone'] == 'VDSR':
    if isinstance(params['datasets'], dict):
        params['datasets']['input_large'] = True
    else:
        params['datasets'] = {'name': params['datasets'], 'input_large': True}

default_keys = {
    'loss': 'L1',
    'self_ensemble': False,
}
params = {**default_keys, **params}

model = SR_LightModel(params).cuda()
logger = TensorBoardLogger("../logs", name='super_resolution')
from foundations import set_tensorboard_logdir
set_tensorboard_logdir(f'../logs/super_resolution')

checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='validation/PSNR', mode='max', prefix=str(params["seed"]))
t_params = get_trainer_params(params)
trainer = Trainer(logger=logger, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=100, **t_params)
trainer.fit(model)

if checkpoint_callback.best_model_path != "":
    from foundations import log_metric, save_artifact
    import numpy as np
    # save_artifact(checkpoint_callback.best_model_path, key='best_model_checkpoint')
    log_val = checkpoint_callback.best_model_score.item()
    log_metric("val_PSNR", float(np.clip(log_val, -1e10, 1e10)))

print("Training finished")
