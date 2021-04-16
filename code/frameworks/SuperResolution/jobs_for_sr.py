import sys
sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.foundation_tools import submit_jobs, random_params


def params_for_SR():
    params = {
        'project_name': 'DIV2Kx2_search_no_large_skip',
        'gpus': 1,
        'num_epochs': 20,
        'weight_decay': 0,
        # 'max_lr': [1e-3, 5e-4, 2e-4, 1e-4],
        'max_lr': 5e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'backbone': {
            'arch': 'inn_sr',
            'version': 'new_spade_act',
            'norm_type': ['spade', 'in', 'bn'],
            'block_skip': False,
            'add_ori': [True, False],
            'use_act': [True, False],
            'add_fea': [True, False],
            'use_esa': [True, False],
            'sub_blocks': [1, 2, 3, 4, 5, 6],
            'num_modules': [1, 2, 3, 4, 5, 6],
            'nf': [32, 50, 64, 128]
        },
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": [233, 234],
    }

    if params['dataset']['name'] == 'DIV2K':
        params['dataset']['test_bz'] = 1
    params['dataset']['scale'] = params['scale']
    params['backbone']['scale'] = params['scale']
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_SR, 'frameworks/SuperResolution/train_sr_model.py', number_jobs=300, job_directory='.')
