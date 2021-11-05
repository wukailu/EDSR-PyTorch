import sys, os
sys.path = [os.getcwd()] + sys.path

from utils.tools import submit_jobs, random_params

# Plain-S 64 width 8 resblock
# Plain-M 64 width 16 resblock
# Plain-L 100 width 16 resblock

pretrain_paths = {
    'resnet20x4': "/data/pretrained/lightning_models/layerwise_resnet20x4_cifar100_b8242.ckpt",
    'resnet20': "/data/pretrained/lightning_models/layerwise_resnet20_cifar100_400ba.ckpt",
    "EDSR50_newtail_short_x2": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_3fa19.ckpt",  # 1000 epoch
    "EDSR50_newtail_short_x4": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_dbb90.ckpt",  # 1000 epoch
    "EDSR64_newtail_short_x2": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_9b790.ckpt",  # 1000 epoch
    "EDSR64_newtail_short_x3": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_27695.ckpt",  # 1000 epoch
    "EDSR64_newtail_short_x4": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_1980c.ckpt",  # 1000 epoch
    "EDSR64_newtail_x2": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_537c4.ckpt",  # 1000 epoch
    "EDSR64_newtail_x3": "/data/pretrained/lightning_models/layerwise_edsrx3_div2k_fe594.ckpt",  # 1000 epoch
    "EDSR64_newtail_x4": "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_69068.ckpt",  # 1000 epoch
    "EDSR100_newtail_x2": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_b00e1.ckpt",  # 1000 epoch
    "EDSR100_newtail_x3": "/data/pretrained/lightning_models/layerwise_edsrx3_div2k_613be.ckpt",  # 1000 epoch
    "EDSR100_newtail_x4": "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_5e9dd.ckpt",  # 1000 epoch
    "EDSR64x2": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_fa9af.ckpt",  # 1000 epoch
    "EDSR100x2": "/data/pretrained/lightning_models/layerwise_edsrx2_div2k_1c96e.ckpt",  # 1000 epoch
    "EDSR64x4": "/data/pretrained/lightning_models/layerwise_edsr64x4_div2k_cbe41.ckpt",  # 1000 epoch
    "EDSRx4": "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_e324f.ckpt",
    "EDSR_100x4": "/data/pretrained/lightning_models/layerwise_edsr100x4_div2k_8b9b5.ckpt",
    "EDSR_200x4": "/data/pretrained/lightning_models/layerwise_edsr200x4_div2k_ca503.ckpt",
    "RDNx4": "/data/pretrained/lightning_models/layerwise_rdnx4_div2k_03029.ckpt",
    "RDNx4_0bias": "/data/pretrained/lightning_models/layerwise_rdnx4_div2k_03029_0bias.ckpt",
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
            'total_batch_size': 32,
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
    'DIV2Kx4-EXP': {
        'task': 'super-resolution',
        'loss': 'L1',
        'gpus': 1,
        'teacher_pretrain_path': "to be filled",
        'max_lr': 2e-4,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 1000,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'scale': 4,
            'total_batch_size': 16,
            'patch_size': 192,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': True,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x4',
    },
    'DIV2Kx3-EXP': {
        'task': 'super-resolution',
        'loss': 'L1',
        'gpus': 1,
        'teacher_pretrain_path': "to be filled",
        'max_lr': 2e-4,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 1000,
        'scale': 3,
        "dataset": {
            'name': "DIV2K",
            'scale': 3,
            'total_batch_size': 16,
            'patch_size': 192,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': True,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x3',
    },
    'DIV2Kx2-EXP': {
        'task': 'super-resolution',
        'loss': 'L1',
        'gpus': 1,
        'teacher_pretrain_path': pretrain_paths['EDSR64x2'],
        'max_lr': 2e-4,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 1000,
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'scale': 2,
            'total_batch_size': 16,
            'patch_size': 192,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': True,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x2',
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
        'max_lr': [0.2],  # 0.05 for plainx4, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
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
        'layer_type': 'normal_no_bn',
        'rank_eps': [0.05, 0.1, 0.2, 0.3],
        'max_lr': [0.05, 0.1, 0.2],  # 0.05 for plainx4, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'seed': 233,
    }

    return {**templates['cifar100-classification'], **params}


def deip_CIFAR100_init_new_distill():
    params = {
        'project_name': 'deip_CIFAR100_init_new',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': [1],
        'layer_type': ['normal_no_bn', 'normal'],
        'rank_eps': [0.05, 0.1],
        'max_lr': [0.05, 0.2],  # 0.05 for plainx4, 0.5 for repvgg on 0.05, 0.2 for repvgg on 0.2, 0.3, 0.5
        'distill_coe': [0.3, 0.5],
        'distill_alpha': [0.01, 0.001],
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': ['MSE'],
        },
        'seed': 233,
    }

    return {**templates['cifar100-classification'], **params}


def params_for_SR_new_init():
    params = {
        'project_name': 'deip_SRx4_init_new_Ridge',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'ridge_alpha': [1e-2],
        'teacher_pretrain_path': pretrain_paths["RDNx4"],
        # 'teacher_pretrain_path': pretrain_paths["EDSRx4"],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1, 0.2],
        'max_lr': [2e-4],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_stable_test():
    params = {
        'project_name': 'deip_SRx4_stable_test',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths["EDSR_200x4"],
        'layer_type': ['normal_no_bn_scale', 'normal_no_bn', 'normal'],
        'fix_r': 200,
        'max_lr': [2e-4],
        'seed': [233, 234],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_new_init_std_align():
    params = {
        'project_name': 'deip_SRx4_init_new_align',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths["RDNx4"],
        # 'teacher_pretrain_path': pretrain_paths["EDSR_100x4"],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1, 0.2],
        'max_lr': [2e-4],
        'std_align': 1,
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_new_init_equal_width():
    params = {
        'project_name': 'deip_SRx4_init_new_equal_width',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths['RDNx4'],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1],
        'fix_r': [50, 70, 100],
        'max_lr': [2e-4],
        'seed': 233,
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_new_init_distill():
    params = {
        'project_name': 'deip_SRx4_init_distill_verify',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths['EDSRx4'],
        'layer_type': ['normal_no_bn'],
        'rank_eps': [0.1],
        'ridge_alpha': 0,
        'max_lr': [2e-4],
        'decompose_adjust': [0, 3],
        'distill_coe': [0.1, 0.3],
        'distill_alpha': [1e-5],
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': ['MSE'],
        },
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_SR_new_init_distill_new_coe():
    params = {
        'project_name': 'deip_SRx4_distill_new_coe',
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths['RDNx4'],
        'layer_type': ['normal_no_bn'],
        'distill_coe_mod': 'new',
        'distill_coe': [0.1, 1, 10],
        'distill_alpha': 1e-5,
        'ridge_alpha': 1e-2,
        'rank_eps': [0.1],
        'max_lr': [2e-4],
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': ['MSE'],
        },
        "fix_distill_module": 1,
        'seed': [233, 234],
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
        'max_lr': [2e-4, 5e-4, 1e-3],
    }

    return {**templates['DIV2K-SRx4'], **params}


def params_for_EXP_main_x2():
    params = {
        'project_name': 'CVPR_EXP_MAIN_x2',
        'method': 'DEIP_Init',
        'fix_r': 100,
        # 'eps': 0.13,
        # 'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x2'],
        'teacher_pretrain_path': pretrain_paths['EDSR100_newtail_x2'],
        'init_stu_with_teacher': 1,
        'layer_type': 'normal_no_bn',
        'ridge_alpha': 0,
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': 'MSE',
        },
        # 'gpus': 2,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def params_for_EXP_main_x3():
    params = {
        'project_name': 'CVPR_EXP_MAIN_x3',
        'method': 'DEIP_Init',
        'fix_r': 100,
        # 'eps': [0.13, 0.12],
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths['EDSR100_newtail_x3'],
        'layer_type': 'normal_no_bn',
        'ridge_alpha': 0,
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': 'MSE',
        },
        'gpus': 2,
        'seed': [233, 234, 235]
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def params_for_EXP_main_x4():
    params = {
        'project_name': 'CVPR_EXP_MAIN_x4',
        'method': 'DEIP_Init',
        'fix_r': 100,
        'init_stu_with_teacher': 1,
        'teacher_pretrain_path': pretrain_paths['EDSR100_newtail_x4'],
        'layer_type': 'normal_no_bn',
        'ridge_alpha': 0,
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': 'MSE',
        },
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def params_for_EXP_ablation_x4():
    params = {
        'project_name': 'CVPR_EXP_Ablation_x4',
        'method': 'DEIP_Init',
        'fix_r': 64,
        'init_stu_with_teacher': [0, 1],
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x4'],
        'layer_type': 'normal_no_bn',
        'ridge_alpha': 0,
        'distill_coe': [0, 0.3],
        'init_distill': [0, 1],
        'decompose_adjust': [0, 3],
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'BridgeDistill',
            'distill_loss': 'MSE',
        },
        'seed': 236,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def params_for_EXP_cmp_init():
    params = {
        'project_name': 'CVPR_EXP_Ablation_Init_x2',
        'method': 'DirectTrain',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x2'],
        'init_stu_with_teacher': 0,
        # 'layer_type': 'repvgg',
        'layer_type': 'normal_no_bn',
        'conv_init': ['kaiming_normal', 'kaiming_uniform', 'xavier_uniform', 'xavier_normal'],
        'gpus': 1,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def params_for_EXP_cmp_repvggx4():
    params = {
        'project_name': 'CVPR_EXP_Ablation_repvgg_x4',
        'method': 'DirectTrain',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x4'],
        'init_stu_with_teacher': 0,
        'layer_type': 'repvgg_no_bn',
        'gpus': 1,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def params_for_EXP_cmp_repvggx3():
    params = {
        'project_name': 'CVPR_EXP_Ablation_repvgg_x3',
        'method': 'DirectTrain',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x3'],
        'init_stu_with_teacher': 0,
        'layer_type': 'repvgg',
        'gpus': 1,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def params_for_EXP_cmp_repvggx2():
    params = {
        'project_name': 'CVPR_EXP_Ablation_repvgg_x2',
        'method': 'DirectTrain',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x2'],
        'init_stu_with_teacher': 0,
        'layer_type': 'repvgg',
        'gpus': 1,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def params_for_EXP_cmp_fakdx4():
    params = {
        'project_name': 'CVPR_EXP_Ablation_FAKD_x4',
        'method': 'Distillation',
        'fix_r': 64,
        # 'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x4'],
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_short_x4'],
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'FAKD',
            # 'position': (0, 16, 33),
            'position': (0, 8, 17),  # for plain-s
        },
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def params_for_EXP_cmp_fakdx3():
    params = {
        'project_name': 'CVPR_EXP_Ablation_FAKD_x3',
        'method': 'Distillation',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x3'],
        # 'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_short_x3'],
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'FAKD',
            'position': (0, 16, 33),  # for plain-m, plain-L
            # 'position': (0, 8, 17),  # for plain-s
        },
        'gpus': 2,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def params_for_EXP_cmp_fakdx2():
    params = {
        'project_name': 'CVPR_EXP_Ablation_FAKD_x2',
        'method': 'Distillation',
        'fix_r': 64,
        # 'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x2'],
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_short_x2'],
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'FAKD',
            # 'position': (0, 33),
            'position': (0, 17),  # for plain-s
        },
        'gpus': 2,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def params_for_EXP_cmp_srkdx4():
    params = {
        'project_name': 'CVPR_EXP_Ablation_SRKD_x4',
        'method': 'Distillation',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x4'],
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'SRKD',
            'position': (0, 16, 33),
        },
        'gpus': 1,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def params_for_EXP_cmp_srkdx3():
    params = {
        'project_name': 'CVPR_EXP_Ablation_SRKD_x3',
        'method': 'Distillation',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x3'],
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'SRKD',
            'position': (0, 16, 33),
        },
        'gpus': 2,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def params_for_EXP_cmp_srkdx2():
    params = {
        'project_name': 'CVPR_EXP_Ablation_SRKD_x2',
        'method': 'Distillation',
        'fix_r': 64,
        'teacher_pretrain_path': pretrain_paths['EDSR64_newtail_x2'],
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'distill_coe': 0.3,
        'distill_alpha': 1e-5,
        'dist_method': {
            'name': 'SRKD',
            'position': (0, 16, 33),
        },
        'gpus': 2,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def test_model():
    params = {
        'project_name': 'model_test',
        'save_model': False,
        'skip_train': True,
        'test_benchmark': True,
        'inference_statics': True,
        # 'load_from': ['/data/tmp/plainmx3_012.ckpt',
        #               '/data/tmp/plainmx3_013.ckpt', ],
        'load_from': ['/data/tmp/fakdx4_233.ckpt', ],
        'width': 0,
        'seed': 233,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def reassess_jobs():
    import torch
    from utils import tools
    import random
    job_filter = {
        "project_name": 'CVPR_EXP_Ablation_x4',
    }
    targets = tools.get_targets((tools.dict_filter(job_filter)))
    print('len targets = ', len(targets))
    random.shuffle(targets)
    t = targets[0]
    ckpt = tools.get_artifacts(t, name='epoch*.ckpt')
    old_params = torch.load(ckpt, map_location=torch.device('cpu'))['hyper_parameters']
    new_params = {**old_params,
                  'save_model': False,
                  'skip_train': True,
                  'test_benchmark': True,
                  'inference_statics': True,
                  'load_from': ckpt,
                  }
    return new_params

def params_for_deip():
    # params = params_for_EXP_main_x2()  # submitted to 233 with 64 width and 2e-4,5e-4 lr and 100 epoch small test
    # params = params_for_EXP_main_x3()  # submitted to 46 with width 100 and eps0.12, 0.13 and to 13 with width 64 and eps 0.13,0.12
    # params = params_for_EXP_main_x4()  # submitted to 236 with width 64 and eps 0.13
    # params = params_for_EXP_ablation_x4()  # submitted to 13, 17, 30, 236 with seed 233, 234 and width 75, 64

    # params = params_for_EXP_cmp_init()

    # params = params_for_EXP_cmp_fakdx4()
    # params = params_for_EXP_cmp_fakdx3()
    # params = params_for_EXP_cmp_fakdx2()

    # params = params_for_EXP_cmp_repvggx2()
    # params = params_for_EXP_cmp_repvggx3()
    # params = params_for_EXP_cmp_repvggx4()

    params = params_for_EXP_cmp_srkdx4()
    # params = params_for_EXP_cmp_srkdx3()
    # params = params_for_EXP_cmp_srkdx2()

    # params = test_model()
    # params = reassess_jobs()
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_deip, 'frameworks/distillation/train_deip_model.py', number_jobs=100, job_directory='.')
