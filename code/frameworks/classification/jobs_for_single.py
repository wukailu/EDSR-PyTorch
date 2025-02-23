import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
sys.path.append('/home/wukailu/EDSR-PyTorch/code/')
from utils.tools import submit_jobs, random_params


def params_for_single_train():
    params = {
        'project_name': 'layerwise_model_train',
        'gpus': 1,
        'num_epochs': 300, 
        'weight_decay': 5e-4,
        'max_lr': [0.2],
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        # 'backbone': ['resnet20_layerwise'],
        'backbone': ['resnet20x4_layerwise', 'resnet20x2_layerwise', 'resnet32_layerwise'],
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": [233, 234, 235, 236],
        'save_model': True,
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
        'save_model': False,
    }
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_single_train, 'frameworks/singlemodel/train_single_model.py', number_jobs=100,
                job_directory='.')
