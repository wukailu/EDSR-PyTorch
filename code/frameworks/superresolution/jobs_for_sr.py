import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
sys.path.append('/home/wukailu/EDSR-PyTorch/code/')

from utils.tools import submit_jobs, random_params
from frameworks.distillation.jobs_for_deip import pretrain_paths

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
        'save_model': True,
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
        'save_model': True,
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
        'save_model': True,
        'inference_statics': True,
        'test_benchmark': True,
        'ignore_exist': True,
        'metric': 'psnr_gray_shave_x4',
    },
}


def test_x2_x2_to_x4():
    pretrained_paths = [
        '/data/kailu/.foundations/job_data/archive/9cc232ba-3760-408d-a89c-7915c2729002/user_artifacts/235epoch=297.ckpt',
        '/data/kailu/.foundations/job_data/archive/7688a2f0-c1c6-4d30-a8d8-ab255b6b8a6b/user_artifacts/235epoch=296.ckpt',
        '/data/kailu/.foundations/job_data/archive/a24082d9-fffb-4647-af7e-73349d46ea08/user_artifacts/235epoch=296.ckpt',
        '/data/kailu/.foundations/job_data/archive/04999c2a-c7bf-4b3e-af5b-87db03a6c810/user_artifacts/235epoch=296.ckpt',
        '/data/kailu/.foundations/job_data/archive/6fa43117-524a-401c-a8b0-55a02299bc98/user_artifacts/235epoch=296.ckpt',
        '/data/kailu/.foundations/job_data/archive/fd0286a6-c865-44e2-b17b-d6bc2f9bb407/user_artifacts/235epoch=296.ckpt',
        '/data/kailu/.foundations/job_data/archive/744f1437-6e55-421b-9a57-5105f8bf281d/user_artifacts/235epoch=296.ckpt',
        '/data/kailu/.foundations/job_data/archive/2161ed64-80a7-4a8a-b77f-ce1a9cdd1351/user_artifacts/235epoch=297.ckpt',
    ]

    params = {
        'project_name': 'x2_x2_to_x4',
        'method': 'TwoStageSR',
        'two_stage_no_freeze': True,
        'num_epochs': 100,
        'pretrained_from': pretrained_paths,
        'backbone': {},
        "seed": 235,
    }

    return {**templates['DIV2K-b16-SRx4'], **params}


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
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def strong_EDSR_x2():
    params = {
        'project_name': 'DIV2Kx4_EXP_EDSRx2_shortmodel',
        'save_model': True,
        'init_from': None,
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': [64, 100],
            # 'n_resblocks': 16,
            'n_resblocks': 8,
        },
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def strong_EDSR_x3():
    params = {
        'project_name': 'DIV2Kx4_EXP_EDSRx3',
        'save_model': True,
        'init_from': pretrain_paths['EDSR64x2'],
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 16,
        },
        'max_lr': 2e-4,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def EDSR_new_tail():
    params = {
        'project_name': 'DIV2Kx4_EDSR_newtail',
        'save_model': True,
        'init_from': None,
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            # 'n_feats': [64],
            'n_feats': [100],
            'n_resblocks': 16,
            'simple_tail': 1,
        },
        'max_lr': 2e-4,
        'seed': [233],
        'gpus': 4,
    }

    return {**templates['DIV2K-b32-SRx4'], **params}


def EDSR_new_tailx2():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': True,
        'init_from': None,
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': [64],
            # 'n_feats': [100],
            'n_resblocks': 16,
            'simple_tail': 1,
        },
        'seed': [233],
        'gpus': 2,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def EDSR_new_tailx3():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': True,
        'init_from': pretrain_paths["EDSR100_newtail_x2"],
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 100,
            'n_resblocks': 16,
            'simple_tail': 1,
        },
        'seed': [233],
        'gpus': 4,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def EDSR_new_tailx4():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': True,
        'init_from': pretrain_paths["EDSR64_newtail_x2"],
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 16,
            'simple_tail': 1,
        },
        'seed': [233],
        'gpus': 1,
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


def naiveBaseline():
    params = {
        'project_name': 'distill_baseline',
        'num_epochs': 1,
        'backbone': {'arch': 'DirectScale_sr'},
        'skip_train': True,
    }

    return {**templates['DIV2K-b512-SRx4'], **params}


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


def PlainFlopsPSNRCurve():
    # resource_per_flops = (16384 / 25)
    # flops = random_params([75, 200, 1350, 3540])
    # resource = resource_per_flops * flops
    # width = random_params([48, 96, 144])
    # depth = int(resource / width / width)

    # depth, width = random_params([(12, 64), (16, 55), (20, 50),
    #                               (16, 100), (20, 87), (24, 73),
    #                               (40, 150), (34, 161), (30, 175),
    #                               (45, 226), (40, 240), (36, 256)])

    depth, width = random_params([(12, 64), (20, 87), (34, 161), (40, 240)])

    params = {
        'project_name': 'plain_SR_curve_b32_new_init',
        'backbone': {
            'arch': 'Plain_layerwise_sr',
            'num_modules': depth,
            'n_feats': width,
            'add_ori': 1,
        },
        'max_lr': 5e-4,
    }

    # return {**templates['DIV2K-b16-SRx4'], **params}
    return {**templates['DIV2K-b32-SRx4'], **params}


def stack_out_test():
    resource = 16384 * 3
    depth = random_params([16, 32, 48])
    width = int((resource / depth) ** 0.5)
    params = {
        'project_name': 'plain_SR_direct_train_300',
        'backbone': {
            'arch': 'Plain_layerwise_sr',
            'num_modules': depth,
            'n_feats': width,
            'add_ori': 1,
            'stack_output': [1, 0],
        },
        'max_lr': [2e-4, 5e-4, 1e-3, 3e-2],
        'seed': [233, 234],
    }

    return {**templates['DIV2K-b32-SRx4'], **params}


def square_test():
    resource = 16384 * 3
    width = random_params([55])
    depth = int(resource / width / width)
    params = {
        'project_name': 'plain_SR_square_test_300',
        'backbone': {
            'arch': 'Plain_layerwise_sr',
            'num_modules': depth,
            'n_feats': width,
            'add_ori': 1,
            'stack_output': 0,
            'square_ratio': [0.1, 0.5],
            'square_num': [1, 2, 4],
            'square_layer_strategy': [0, 1, 2],
            'square_before_relu': [1],
        },
        'num_epochs': 300,
    }

    return {**templates['DIV2K-b32-SRx4'], **params}


def bn_test():
    resource = 16384 * 3
    width = random_params([55])
    depth = int(resource / width / width)
    params = {
        'project_name': 'plain_SR_bn_test_300',
        'backbone': {
            'arch': 'Plain_layerwise_sr',
            'num_modules': depth,
            'n_feats': width,
            'add_ori': 1,
            'stack_output': 0,
            'layerType': 'normal',
        },
        'num_epochs': 300,
        'max_lr': [2e-3, 5e-3, 1e-2, 2e-2],
    }

    return {**templates['DIV2K-b32-SRx4'], **params}


def EDSRx2x2_to_x4():
    params = {
        'project_name': 'EDSRx2x2=x4',
        'method': 'TwoStageSR',
        'two_stage_no_freeze': True,
        'num_epochs': 300,
        'pretrained_from': pretrain_paths['EDSR64x2'],
        'backbone': {},
        "seed": 233,
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def Short_EDSR_new_tailx2():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': True,
        'init_from': None,
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 8,
            'simple_tail': 1,
        },
        'seed': [233],
        'gpus': 4,
    }

    return {**templates['DIV2Kx2-EXP'], **params}


def Short_EDSR_new_tailx3():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': True,
        'init_from': pretrain_paths['EDSR64_newtail_short_x2'],
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 8,
            'simple_tail': 1,
        },
        'seed': [233],
        'gpus': 8,
    }

    return {**templates['DIV2Kx3-EXP'], **params}


def Short_EDSR_new_tailx4():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': True,
        'init_from': pretrain_paths['EDSR64_newtail_short_x2'],
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 64,
            'n_resblocks': 8,
            'simple_tail': 1,
        },
        'seed': [233],
    }

    return {**templates['DIV2Kx4-EXP'], **params}


def test_model():
    params = {
        'project_name': 'EDSR_newtail',
        'save_model': False,
        'skip_train': True,
        'init_from': pretrain_paths['EDSR100_newtail_x4'],
        'backbone': {
            'arch': ['EDSR_layerwise_sr'],
            'n_feats': 100,
            'n_resblocks': 16,
            'simple_tail': 1,
        },
        'seed': [233],
    }

    return {**templates['DIV2Kx4-EXP'], **params}



def params_for_SR():
    # 所有实验基础模型, 64 宽度一版 100 宽度一版
    # params = EDSR_new_tailx2()  # finished
    # params = EDSR_new_tailx3()  # finished
    # params = EDSR_new_tailx4()  # finished

    # 与 PISR, RepVGG 等比较使用的模型
    # params = Short_EDSR_new_tailx2()  # finished
    # params = Short_EDSR_new_tailx3()  # submitted to 236 with width 64
    # params = Short_EDSR_new_tailx4()  #

    # params = directTrainPlain()
    # params = dense_model_train()

    params = test_model()

    params = random_params(params)
    if 'scale' not in params['backbone']:
        params['backbone']['scale'] = params['scale']
    return params


if __name__ == "__main__":
    submit_jobs(params_for_SR, 'frameworks/superresolution/train_sr_model.py', number_jobs=1000, job_directory='.')
