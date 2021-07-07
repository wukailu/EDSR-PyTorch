import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.tools import submit_jobs, random_params


def params_for_single_train():
    params = {
        'project_name': 'deip_test',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 5e-4,
        'max_lr': [0.1, 0.2, 0.3],
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': ['resnet20_act_wise'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }
    return random_params(params)


def params_for_test():
    params = {
        'project_name': 'test',
        'gpus': 1,
        'num_epochs': 1,
        'weight_decay': 5e-4,
        'max_lr': 0.1,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': ['resnet20'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_test, 'frameworks/singlemodel/train_single_model.py', number_jobs=1,
                job_directory='.')
