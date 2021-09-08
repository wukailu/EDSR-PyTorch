import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.tools import submit_jobs, random_params

pretrain_paths = {
    'resnet20x4': "/data/pretrained/lightning_models/layerwise_resnet20x4_cifar100_b8242.ckpt",
    'resnet20': "/data/pretrained/lightning_models/layerwise_resnet20_cifar100_400ba.ckpt",
    "EDSRx4": "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_a0131.ckpt",
}

templates = {
    "cifar100-classification": {
        'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 5e-4,
        'max_lr': 0.2,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': pretrain_paths['resnet20x4'],
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
            'batch_size': 32,
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
    },
}


def params_for_SR_real_progressive():
    params = {
        'project_name': 'deip_SRx4_freeze_progressive',
        'method': 'DEIP_Full_Progressive',
        'description': 'DEIP_Full_Progressive',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal_no_bn'],
        'lr_scheduler': ['OneCycLR', 'none'],
        'rank_eps': [0.3],
        'freeze_trained': [0, 1],
        'freeze_teacher_bn': [0, 1],
        'max_lr': [2e-4],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_real_progressive_small():
    params = {
        'project_name': 'deip_SRx4_progressive_small',
        'num_epochs': 100,
        'max_lr': [2e-4],
        'seed': [233, 234],
        'gpus': 1,
    }

    return {**params_for_SR_real_progressive(), **params}


def params_for_SR_progressive():
    params = {
        'project_name': 'deip_SRx4_progressive',
        'method': 'Progressive_Distillation',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.05],
        'max_lr': [5e-4],
        'distill_coe': [0, 1, 3, 10],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_progressive_small():
    params = {
        'project_name': 'deip_SRx4_progressive_small',
        'method': 'Progressive_Distillation',
        'layer_type': 'normal_no_bn',
        'init_stu_with_teacher': [1],
        'num_epochs': 100,
        'max_lr': [5e-4],
        'distill_coe': [0, 0.01, 1e-4, 1],
        'gpus': 1,
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


def params_for_SR_baseline_with_add_ori():
    params = {
        'project_name': 'deip_SRx4_baseline',
        'add_ori': [1],
        'init_stu_with_teacher': [0],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.01, 0.05, 0.1, 0.2],  # 0.05, 0.6, 1, 2
        'max_lr': [2e-4, 5e-4],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_baseline():
    params = {
        'project_name': 'deip_SRx4_baseline',
        'description': 'direct_train',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.01, 0.05, 0.1, 0.2],  # 0.05, 0.6, 1, 2
        'max_lr': [2e-4, 5e-4],
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
        'project_name': 'deip_baselines_20x4',
        'description': 'direct_train',
        'layer_type': 'normal',
        'init_stu_with_teacher': [0],
        # 'rank_eps': [0.01, 0.05, 0.1, 0.2],
        'rank_eps': [0.3, 0.4, 0.5],
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
        'layer_type': 'normal',
        'init_stu_with_teacher': [1],
        'rank_eps': [0.01, 0.05, 0.1, 0.2],
        'max_lr': [0.2],
        'seed': 233,
    }

    return {**templates['cifar100-classification'], **params}


def params_for_unit_test():
    params = {
        'project_name': 'unit_test',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1],  # 0.05, 0.6, 1, 2
        'max_lr': [5e-4],
        'num_epochs': 1,
        'seed': 0,
    }

    return {**templates['DIV2K-SRx4'], **params}


def deip_CIFAR100_init_new():
    params = {
        'project_name': 'deip_CIFAR100_init_new',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': [0, 1],
        'layer_type': 'normal',
        'rank_eps': [0.05, 0.1, 0.2, 0.3],
        'max_lr': [0.05, 0.1, 0.2],  # 0.05 for plane, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
    }

    return {**templates['cifar100-classification'], **params}


def params_for_SR_new_init():
    params = {
        'project_name': 'deip_SRx4_init_new',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1, 0.2, 0.3],
        'max_lr': [1e-4, 2e-4, 5e-4],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_new_conv_init():
    params = {
        'project_name': 'deip_SRx4_init_new_conv_init',
        'description': 'trying to avoid vanishing',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 0,
        'init_tail': 1,
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1],
        'max_lr': [5e-5, 1e-4, 2e-4, 5e-4],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_deip():
    # params = params_for_baseline()
    # params = params_for_deip_distillation()
    # params = params_for_deip_progressive_distillation()
    # params = deip_CIFAR100_init_new()

    # params = params_for_SR_baseline()
    # params = params_for_SR_baseline_small()
    # params = params_for_SR_init()
    # params = params_for_SR_structure()
    # params = params_for_SR_progressive()
    # params = params_for_SR_progressive_small()
    # params = params_for_SR_real_progressive()
    # params = params_for_SR_real_progressive_small()
    # params = params_for_SR_baseline_with_add_ori()
    params = params_for_SR_new_init()

    # params = params_for_unit_test()
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=100, job_directory='.')
    # submit_jobs(params_for_deip, 'frameworks/distillation/unit_test.py', number_jobs=1, job_directory='.')
