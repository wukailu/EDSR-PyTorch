import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.tools import submit_jobs, random_params

pretrain_paths = {
    'resnet': "/data/pretrained/lightning_models/layerwise_resnet20_cifar100_58603.ckpt",
}


def params_for_direct_train():
    params = {
        'project_name': 'deip_initialization',
        'description': 'direct_train',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal', 'repvgg'],
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [0.05],  # 5e-2, 0.3
        'weight_decay': 5e-4,
        'max_lr': [0.05, 0.2, 0.5],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'ignore_exist': True,
    }

    return params


def params_for_deip_distillation():
    params = {
        'project_name': 'deip_distillation_repeat',
        'description': 'common_distillation',
        'method': 'Distillation',
        'dist_method': ['CKA_on_logits', 'KD', 'Progressive_FD', 'FD_Conv1x1_MSE', 'KA_on_channel'],
        'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': 5e-2,
        'distill_coe': [1, 0.1, 0.01],
        'weight_decay': 5e-4,
        'max_lr': [0.2, 0.5],
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
        'project_name': 'deip_distillation_repeat',
        'description': 'progressive_distillation',
        'method': 'Progressive_Distillation',
        'gpus': 1,
        'num_epochs': 300,
        'track_grad_norm': True,
        'rank_eps': 5e-2,
        'distill_coe': [1, 0.1, 0.01],
        'weight_decay': 5e-4,
        'max_lr': [1e-4, 1e-3, 1e-2, 1e-1],
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }

    return params


def params_for_deip():
    params = params_for_direct_train()
    # params = params_for_deip_distillation()
    # params = params_for_deip_progressive_distillation()

    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=1000, job_directory='.')
