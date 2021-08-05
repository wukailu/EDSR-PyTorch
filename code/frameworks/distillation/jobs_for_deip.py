import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.tools import submit_jobs, random_params

pretrain_paths = {
    'resnet': "/data/pretrained/lightning_models/layerwise_resnet20_cifar100_58603.ckpt",
    "EDSRx4": "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_fc971.ckpt",
}

templates = {
    "cifar100-classification": {
        'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 5e-4,
        'max_lr': [0.2],
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'ignore_exist': True,
        'save_model': False,
    },
    'DIV2K-SRx4': {
        'task': 'super-resolution',
        'loss': 'L1',
        'metric': 'psnr255',
        'rgb_range': 255,
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'teacher_pretrain_path': pretrain_paths['EDSRx4'],
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            "test_bz": 1,
            'scale': 4,
        },
        'scale': 4,
        "seed": [233, 234, 235, 236],
        'ignore_exist': True,
        'save_model': False,
    }
}


def params_for_SR_progressive():
    params = {
        'project_name': 'deip_SRx4_progressive',
        'description': 'progressive_distillation',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal_no_bn', 'plain_sr'],
        'rank_eps': [0.05],
        'max_lr': [1e-4, 2e-4, 5e-4, 1e-3],
        'distill_coe': [1e-3, 1e-4, 0],
    }

    return {**templates['DIV2K-SRx4'], **params}

def params_for_SR_progressive_small():
    params = {
        'project_name': 'deip_SRx4_progressive_small',
        'init_stu_with_teacher': [1],
        'num_epochs': 100,
        'max_lr': [1e-4, 2e-4, 5e-4, 1e-3],
        'distill_coe': [1e-3, 1e-4, 0],
    }

    return {**params_for_SR_progressive(), **params}


def params_for_SR_structure():
    params = {
        'project_name': 'deip_SRx4_structure',
        'description': 'direct_train',
        'init_stu_with_teacher': [0],
        'layer_type': 'plain_sr-bn',  # ['plain_sr', 'plain_sr-bn', 'plain_sr-prelu'],
        'rank_eps': [0.05],
        'max_lr': [2e-4, 5e-4, 1e-3],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_init():
    params = {
        'project_name': 'deip_SRx4_init_rerun',
        'description': 'direct_train',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal', 'normal_no_bn', 'normal_no_bn_prelu'],
        'rank_eps': [0.1],  # 0.05, 0.6, 1, 2
        'max_lr': [5e-4, 1e-3],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_baseline():
    params = {
        'project_name': 'deip_SRx4_baseline',
        'description': 'direct_train',
        'init_stu_with_teacher': [0],
        'layer_type': ['normal', 'normal_no_bn', 'normal_no_bn_prelu'],
        'rank_eps': [0.05],  # 0.05, 0.6, 1, 2
        'max_lr': [5e-4, 1e-3, 3e-3],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_baseline_small():
    params = params_for_SR_baseline()
    params['num_epochs'] = 100
    params['project_name'] = 'deip_SRx4_baseline100'
    params['max_lr'] = 5e-4
    return params


def params_for_baseline():
    params = {
        'project_name': 'deip_baselines',
        'description': 'direct_train',
        'init_stu_with_teacher': [0],
        'rank_eps': [3],  # 0.05, 0.6, 1, 2
        'max_lr': [0.2],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
    }

    return {**templates['cifar100-classification'], **params}


def params_for_direct_train():
    params = {
        'project_name': 'deip_initialization',
        'description': 'direct_train',
        'init_stu_with_teacher': [1],
        'rank_eps': [1, 2],  # 0.05, 0.6, 1, 2
        'max_lr': [0.2],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
    }

    return {**templates['cifar100-classification'], **params}


def params_for_deip_distillation():
    params = {
        'project_name': 'deip_distillation_repeat',
        'description': 'common_distillation',
        'init_stu_with_teacher': [0, 1],
        'method': 'Distillation',
        'dist_method': ['KD', 'Progressive_FD', 'FD_Conv1x1_MSE'],
        'rank_eps': 5e-2,
        'distill_coe': [1, 0.1],
        'max_lr': [0.2],
    }

    return {**templates['cifar100-classification'], **params}


def params_for_deip_progressive_distillation():
    params = {
        'project_name': 'deip_distillation_progressive',
        'description': 'progressive_distillation',
        'method': 'Progressive_Distillation',
        'init_stu_with_teacher': [1],
        'rank_eps': [2],
        'distill_coe': [1e-3, 1e-4, 0],
        'max_lr': [0.2],
    }

    return {**templates['cifar100-classification'], **params}


def params_for_deip():
    # params = params_for_baseline()
    # params = params_for_direct_train()
    # params = params_for_deip_distillation()
    # params = params_for_deip_progressive_distillation()

    # params = params_for_SR_baseline()
    # params = params_for_SR_baseline_small()
    # params = params_for_SR_init()
    # params = params_for_SR_structure()
    params = params_for_SR_progressive()

    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=100, job_directory='.')
