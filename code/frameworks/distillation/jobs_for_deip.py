import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.tools import submit_jobs, random_params

pretrain_paths = {
    'resnet': "/data/pretrained/lightning_models/layerwise_resnet20_cifar100_58603.ckpt",
    "EDSR": "",
}


def params_for_SR_baseline():
    params = {
        'project_name': 'deip_SR_baselines',
        'description': 'direct_train',
        'init_stu_with_teacher': [0],
        'layer_type': ['repvgg'],
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [0.05],  # 0.05, 0.6, 1, 2
        'weight_decay': 5e-4,
        'max_lr': [2e-4, 2e-3, 2e-2],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
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
        'save_model': False,
        # "seed": [233, 234, 235, 236],
        'ignore_exist': True,
    }

    return params



def params_for_baseline():
    params = {
        'project_name': 'deip_baselines',
        'description': 'direct_train',
        'init_stu_with_teacher': [0],
        'layer_type': ['repvgg'],
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [3],  # 0.05, 0.6, 1, 2
        'weight_decay': 5e-4,
        'max_lr': [0.2],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'ignore_exist': True,
        'save_model': False,
    }

    return params


def params_for_direct_train():
    params = {
        'project_name': 'deip_initialization',
        'description': 'direct_train',
        'init_stu_with_teacher': [1],
        'layer_type': ['repvgg'],
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [1, 2],  # 0.05, 0.6, 1, 2
        'weight_decay': 5e-4,
        'max_lr': [0.2],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'ignore_exist': True,
        'save_model': False,
    }

    return params


def params_for_deip_distillation():
    params = {
        'project_name': 'deip_distillation_repeat',
        'description': 'common_distillation',
        'init_stu_with_teacher': [0, 1],
        'method': 'Distillation',
        'dist_method': ['KD', 'Progressive_FD', 'FD_Conv1x1_MSE'],
        'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': 5e-2,
        'distill_coe': [1, 0.1],
        'weight_decay': 5e-4,
        'max_lr': [0.2],
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'ignore_exist': True,
    }

    return params


def params_for_deip_progressive_distillation():
    params = {
        'project_name': 'deip_distillation_progressive',
        'description': 'progressive_distillation',
        'method': 'Progressive_Distillation',
        'init_stu_with_teacher': [1],
        'layer_type': ['repvgg'],
        'gpus': 1,
        'num_epochs': 300,
        # 'track_grad_norm': True,
        'rank_eps': [2],
        'distill_coe': [1e-3, 1e-4, 0],
        'weight_decay': 5e-4,
        'max_lr': [0.2],
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'ignore_exist': True,
    }

    return params


def params_for_deip():
    # params = params_for_baseline()
    # params = params_for_direct_train()
    params = params_for_deip_distillation()
    # params = params_for_deip_progressive_distillation()

    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=1000, job_directory='.')
