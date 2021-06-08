import sys
sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.foundation_tools import submit_jobs, random_params


def params_for_single_train():
    params = {
        'project_name': 'framework_test',
        'gpus': 1,
        'num_epochs': 2,
        'weight_decay': 5e-4,
        'max_lr': 0.01,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'backbone': ['resnet20'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_single_train, 'frameworks/SingleModel/train_single_model.py', number_jobs=1000, job_directory='.')
