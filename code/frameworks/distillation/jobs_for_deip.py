import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.foundation_tools import submit_jobs, random_params


def params_for_direct_train():
    params = {
        'project_name': 'deip_distillation_repeat',
        'description': 'repvgg_direct',
        'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [5e-2],  # 5e-2
        'weight_decay': 5e-4,
        'max_lr': [0.5],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': 'resnet20_act_wise',
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
    }

    return params


def params_for_deip_distillation():
    params = {
        'project_name': 'deip_distillation_repeat',
        'description': 'distillation_test',
        'method': 'Distillation',
        'dist_method': ['CKA_on_logits', 'KD', 'Progressive_FD', 'FD_Conv1x1_MSE'],  # 'CKA_on_channel'
        'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': 5e-2,
        'distill_coe': [1, 0.1, 0.01, 0.001],
        # 'distill_coe': [0],
        'weight_decay': 5e-4,
        'max_lr': 0.5,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': 'resnet20_act_wise',
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'save_model': False,
    }

    return params


def params_for_deip_progressive_distillation():
    params = {
        'project_name': 'deip_test',
        'description': 'progressive_distillation_with_bug_fix',
        'method': 'Progressive_Distillation',
        'gpus': 1,
        'num_epochs': 4000,
        'track_grad_norm': True,
        # 'rank_eps': [5e-2, 0.2, 0.3, 0.5],  #  5e-2
        'rank_eps': 5e-2,
        # 'distill_coe': [1, 0.1, 0.01],
        'distill_coe': 0,
        'weight_decay': 5e-4,
        'max_lr': [1e-4, 1e-3, 1e-2],
        'optimizer': 'SGD',
        'backbone': 'resnet20_act_wise',
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }

    return params


def params_for_deip():
    params = params_for_deip_distillation()

    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=100, job_directory='.')
