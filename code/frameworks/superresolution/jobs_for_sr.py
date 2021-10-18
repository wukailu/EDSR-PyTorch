import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
sys.path.append('/home/wukailu/EDSR-PyTorch/code/')

from utils.tools import submit_jobs, random_params

templates = {
    'DIV2K-b16-SRx4': {
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 300,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'scale': 4,
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
    },
    'DIV2K-b32-SRx4': {
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 300,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'scale': 4,
            'total_batch_size': 32,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
    },
    'DIV2K-b64-SRx4': {
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 300,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'scale': 4,
            'total_batch_size': 64,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
    },
    'DIV2K-b16-SRx2': {
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 300,
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'scale': 2,
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
    },
    'DIV2K-b512-SRx4': {
        'weight_decay': 0,
        'max_lr': 1e-3,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 300,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'scale': 4,
            'total_batch_size': 512,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
    },
    'DIV2K-b512-SRx2': {
        'weight_decay': 0,
        'max_lr': 1e-3,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'num_epochs': 300,
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'scale': 2,
            'total_batch_size': 512,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
            'test_bz': 1,
        },
        'rgb_range': 255,
        "seed": [233, 234, 235, 236],
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
    },
    'DIV2Kx2-EXP': {
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
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x2',
    },
    'DIV2Kx3-EXP': {
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
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x3',
    },
    'DIV2Kx4-EXP': {
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
        'save_model': False,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x4',
    },
}


def dense_model_train():
    params = {
        'project_name': 'DIV2Kx4_EXP_Baseline',
        'save_model': True,
        'backbone': {
            'arch': ['HAN_sr', 'RFDN_sr', 'RDN_sr', 'SRCNN_sr', 'FSRCNN_sr', 'CARN_sr', 'CARN_M_sr'],
            # 'arch': ['IMDN_sr'],
            # 'arch': ['EDSR_layerwise_sr', 'EDSR_sr'],
            # 'arch': ['RDN_layerwise_sr', 'RDN_sr'],
            # 'arch': ['RDN_layerwise_sr'],
            # 'RDNconfig': 'A',
        },
        'seed': 233,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def strong_EDSR_x2():
    params = {
        'project_name': 'DIV2Kx4_EXP_EDSRx2',
        'save_model': True,
        'init_from': None,
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': [64, 100],
            'n_resblocks': 16,
        },
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def strong_EDSR_x3():
    params = {
        'project_name': 'DIV2Kx4_EXP_EDSRx3',
        'save_model': True,
        'init_from': 'to be filled',
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 16,
        },
        'max_lr': [1e-4, 2e-4, 5e-4],
        'seed': 233,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def strong_EDSR_x4():
    params = {
        'project_name': 'DIV2Kx4_EXP_EDSRx4',
        'save_model': True,
        'init_from': 'to be filled',
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 16,
        },
        'max_lr': [1e-4, 2e-4, 5e-4],
        'seed': 233,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def inference_test():
    params = {
        'project_name': 'inference_benchmark',
        'num_epochs': 1,
        'backbone': {
            'arch': ['EDSR_sr', 'RCAN_sr', 'HAN_sr', 'IMDN_sr', 'RFDN_sr', 'RDN_sr', 'SRCNN_sr', 'FSRCNN_sr', 'CARN_sr',
                     'CARN_M_sr'],
        },
    }
    return {**templates['DIV2K-b16-SRx4'], **params}


def directTrainPlain():
    depth, width = random_params([(20, 87), (10, 114), (14, 100)])
    params = {
        'project_name': 'plain_SR_add_ori_verify',
        'num_epochs': 300,
        'backbone': {
            'arch': 'Plain_layerwise_sr',
            'num_modules': depth,
            'n_feats': width,
            'add_ori': 1,
            'tail': ['easy'],
        },
    }

    return {**templates['DIV2K-b32-SRx4'], **params}




def params_for_SR():
    # params = directTrainPlain()
    # params = dense_model_train()
    # params = inference_test()
    params = strong_EDSR_x2()
    # params = strong_EDSR_x3()
    # params = strong_EDSR_x4()

    params = random_params(params)
    if 'scale' not in params['backbone']:
        params['backbone']['scale'] = params['scale']
    return params


if __name__ == "__main__":
    submit_jobs(params_for_SR, 'frameworks/superresolution/train_sr_model.py', number_jobs=1000, job_directory='.')
