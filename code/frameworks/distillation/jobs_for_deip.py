import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.foundation_tools import submit_jobs, random_params


def params_for_direct_train():
    params = {
        'project_name': 'deip_test',
        'total_batch_size': 256,  # real batch size will be gpus * batch size
        'gpus': 1,
        'num_epochs': 2,
        'weight_decay': 5e-4,
        'max_lr': 0.01,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': 'resnet20_act_wise',
        "dataset": "cifar100",
        "seed": 0,
    }

    return params


def params_for_deip():
    params = params_for_direct_train()

    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=100, job_directory='.')
