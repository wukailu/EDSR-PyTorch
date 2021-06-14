import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.foundation_tools import submit_jobs, random_params


def params_for_direct_train():
    params = {
        'project_name': 'deip_test',
        'description': 'direct_training',
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [5e-2, 0.2, 0.3, 0.5],  #  5e-2
        'weight_decay': 5e-4,
        'max_lr': 0.01,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': 'resnet20_act_wise',
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }

    return params


def params_for_deip_distillation():
    params = {
        'project_name': 'deip_test',
        'description': 'distillation_with_FD_Conv1x1_MSE_and_coe_1',
        'method': 'Distillation',
        'gpus': 1,
        'num_epochs': 300,
        'rank_eps': [5e-2, 0.2, 0.3, 0.5],  #  5e-2
        # 'rank_eps': 5e-2,
        'distill_coe': [1, 2, 4],
        'weight_decay': 5e-4,
        'max_lr': 0.01,
        'lr_scheduler': 'OneCycLR',
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
