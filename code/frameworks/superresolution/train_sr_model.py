import sys
import os

sys.path.append(os.getcwd())

from frameworks.superresolution.SRModel import load_model
from frameworks.singlemodel.train_single_model import get_params, train_model, inference_statics
import utils.backend as backend


def prepare_params(params):
    from utils.tools import parse_params
    params = parse_params(params)
    if params['backbone'] == 'VDSR':
        if isinstance(params['datasets'], dict):
            params['datasets']['input_large'] = True
        else:
            params['datasets'] = {'name': params['datasets'], 'input_large': True}
    default_keys = {
        'metric': 'psnr255',
        'inference_statics': False,
        'skip_train': False,
        'save_model': False,
        'test_benchmark': False,
    }
    params = {**default_keys, **params}
    return params


if __name__ == "__main__":
    params = get_params()
    params = prepare_params(params)
    print(params)

    model = load_model(params)

    if not params['skip_train']:
        model = train_model(model, params, save_name='super_resolution', checkpoint_monitor='validation/psnr255',
                            mode='max')

    if params['test_benchmark']:
        import torch
        from datasets import DataProvider

        benchmarks = ['Set5', 'Set14', 'B100', 'Urban100']
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
                    model.step((x.cuda(), y.cuda(), _), d)
                # for k, v in metrics.items():
                #     log_metric(d + "_" + k, float(np.clip(v.item(), -1e10, 1e10)))

    if params['inference_statics']:
        inference_statics(model, batch_size=1)
