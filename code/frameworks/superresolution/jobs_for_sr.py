import sys

sys.path.append('/home/kailu/EDSR-PyTorch/code/')
from utils.foundation_tools import submit_jobs, random_params


def search_for_plane():
    params = {
        'project_name': 'search_for_plane_x2',
        'gpus': 1,
        'num_epochs': 20,
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'backbone': {
            'arch': 'Plane_sr',
            'nf': [50, 128],
            'num_modules': [1, 2, 3, 4],
            'norm_type': ['spade', 'none'],
            'conv_in_block': [0, 1, 2, 3],
            'use_act': [True, False],
            'norm_before_relu': [False, True],
            # 'norm_before_relu': True,
            'use_esa': [False, True],
            # 'use_esa': False,
            'use_spade': [True, False],
            'large_ori': [False, True],
            # 'large_ori': True,
        },
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": 233,
        'save_model': False
    }

    return params


def get_search_params():
    params = {
        'project_name': 'DIV2Kx2_search_no_large_skip_no_preactivation',
        'gpus': 1,
        'num_epochs': 20,
        'weight_decay': 0,
        # 'max_lr': [1e-3, 5e-4, 2e-4, 1e-4],
        'max_lr': 5e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'backbone': {
            'arch': 'inn_sr',
            'version': 'new_spade_act',
            'norm_type': ['spade', 'in', 'bn', 'none'],
            'block_skip': [True, False],
            'add_ori': [True, False],
            'use_act': [True, False],
            'add_fea': [True, False],
            'use_esa': [True, False],
            'sub_blocks': [1, 2, 3, 4, 5, 6],
            'num_modules': [1, 2, 3, 4, 5, 6],
            'nf': [32, 50, 64, 128]
        },
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": [233, 234],
        'save_model': False
    }

    return params


def get_selected_params():
    backbones = [
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
         'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 1, 'num_modules': 3, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': False,
         'use_act': True, 'add_fea': True, 'use_esa': False, 'sub_blocks': 2, 'num_modules': 3, 'nf': 64},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': False,
         'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 6, 'num_modules': 1, 'nf': 64},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
         'use_act': False, 'add_fea': False, 'use_esa': False, 'sub_blocks': 1, 'num_modules': 3, 'nf': 128,
         'scale': 2},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'bn', 'block_skip': True, 'add_ori': False,
         'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 1, 'num_modules': 6, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': False,
         'use_act': False, 'add_fea': True, 'use_esa': True, 'sub_blocks': 1, 'num_modules': 5, 'nf': 64},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': False,
         'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 1, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
         'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 2, 'nf': 64},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': False,
         'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 1, 'num_modules': 5, 'nf': 64},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
         'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
         'use_act': False, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 1, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
         'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
         'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 4, 'nf': 50},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'in', 'block_skip': False, 'add_ori': False,
         'use_act': False, 'add_fea': False, 'use_esa': True, 'sub_blocks': 3, 'num_modules': 6, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': False,
         'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 64},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
         'use_act': False, 'add_fea': False, 'use_esa': True, 'sub_blocks': 1, 'num_modules': 4, 'nf': 50},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
         'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 1, 'nf': 64},
    ]

    params = {
        'project_name': 'Final_Test',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 0,
        'max_lr': 5e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'backbone': backbones,
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": [233, 234],
    }

    return random_params(params)


def search_for_with_feat():
    params = {
        'project_name': 'DIV2Kx2_search_vgg_feat',
        'gpus': 1,
        'num_epochs': 20,
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'backbone': {
            'arch': 'inn_sr',
            'version': 'new_spade_act',
            'norm_type': ['spade'],
            'block_skip': False,
            'add_ori': True,
            'use_act': True,
            'add_fea': False,
            'use_esa': [True, False],
            'vgg_feat': True,
            'sub_blocks': [1, 2, 3],
            'num_modules': [1, 2],
            'nf': [50, 128]
        },
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": [235],
    }

    return params


def get_final_few():
    backbones = [
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': False,
        #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 64},
        # {'arch': 'rfdn_sr'},
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
        #  'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 1, 'nf': 64},
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
        #  'use_act': False, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 1, 'nf': 128},
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
        #  'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 1, 'nf': 64},
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'bn', 'block_skip': True, 'add_ori': False,
        #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 1, 'num_modules': 6, 'nf': 128},
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
        #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 128},
        {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
         'use_act': False, 'add_fea': False, 'use_esa': False, 'sub_blocks': 1, 'num_modules': 3, 'nf': 128},
        # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
        #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 4, 'nf': 50},
    ]

    params = {
        'project_name': 'Final_Test_on_Benchmark_x4',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 0,
        'max_lr': 2e-4,
        'optimizer': 'Adam',
        'lr_scheduler': 'OneCycLR',
        'backbone': backbones,
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": 235,
        'inference_statics': True,
        'test_benchmark': True,
    }

    return random_params(params)


def final_test_for_plane():
    backbones = [
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': False},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'bn', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'none', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': False, 'use_esa': False, 'use_spade': True, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 1, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': False},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 3, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': False},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'none', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'in', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'bn', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': True},
        {'arch': 'Plane_sr', 'nf': 50, 'num_modules': 4, 'norm_type': 'none', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': False},
        {'arch': 'Plane_sr', 'nf': 50, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': True, 'use_esa': True, 'use_spade': True, 'large_ori': False},
        {'arch': 'Plane_sr', 'nf': 50, 'num_modules': 2, 'norm_type': 'none', 'conv_in_block': 3, 'use_act': False,
         'norm_before_relu': False, 'use_esa': True, 'use_spade': True, 'large_ori': False},
    ]

    params = {
        'project_name': 'Plane_on_Benchmark_x2',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 0,
        'max_lr': 2e-4,
        'optimizer': 'Adam',
        'lr_scheduler': 'OneCycLR',
        'backbone': backbones,
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": 235,
        'inference_statics': True,
        'test_benchmark': True,
    }

    return random_params(params)


def final_test_for_plane_x2():
    backbones = [
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': True,
         'norm_before_relu': True, 'use_esa': True, 'use_spade': False, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': True, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': True,
         'norm_before_relu': True, 'use_esa': True, 'use_spade': True, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 2, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': True, 'use_esa': True, 'use_spade': True, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 3, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': False, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 1, 'use_act': True,
         'norm_before_relu': True, 'use_esa': True, 'use_spade': False, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 1, 'use_act': False,
         'norm_before_relu': True, 'use_esa': False, 'use_spade': False, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
         'norm_before_relu': False, 'use_esa': False, 'use_spade': True, 'large_ori': True, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'spade', 'conv_in_block': 3, 'use_act': True,
         'norm_before_relu': False, 'use_esa': True, 'use_spade': True, 'large_ori': False, 'scale': 2},
        {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 4, 'norm_type': 'none', 'conv_in_block': 2, 'use_act': False,
         'norm_before_relu': False, 'use_esa': True, 'use_spade': True, 'large_ori': False, 'scale': 2},
    ]

    params = {
        'project_name': 'Plane_on_Benchmark_x2',
        'gpus': 1,
        'num_epochs': 300,
        'weight_decay': 0,
        'max_lr': 2e-4,
        'optimizer': 'Adam',
        'lr_scheduler': 'OneCycLR',
        'backbone': backbones,
        'scale': 2,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": 235,
        'inference_statics': True,
        'test_benchmark': True,
    }

    return random_params(params)


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
        'gpus': 1,
        'num_epochs': 100,
        'weight_decay': 0,
        'max_lr': 2e-5,
        'optimizer': 'Adam',
        'lr_scheduler': 'OneCycLR',
        'pretrained_from': pretrained_paths,
        'backbone': {},
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": 235,
        'inference_statics': True,
        'test_benchmark': True,
    }

    return random_params(params)


def dense_model_test():
    params = {
        'project_name': 'DIV2Kx4_model_test',
        'gpus': 1,
        'num_epochs': 30,
        'weight_decay': 0,
        'max_lr': 2e-4,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'backbone': {
            # 'arch': ['EDSR_sr', 'RCAN_sr', 'HAN_sr', 'IMDN_sr', 'RDN_sr'],
            # 'arch': ['RDN_free_sr'],
            'arch': ['IMDN_free_sr'],
            'nf': 50,
        },
        'scale': 4,
        "dataset": {
            'name': "DIV2K",
            'total_batch_size': 16,
            'patch_size': 96,
            'ext': 'sep',
            'repeat': 20,
        },
        'rgb_range': 255,
        "seed": [235],
    }

    return params


def params_for_SR():
    params = dense_model_test()

    # if params['backbone']['norm_type'] == 'spade':
    #     params['max_lr'] = min(params['max_lr'], 2e-4)
    if params['dataset']['name'] == 'DIV2K':
        params['dataset']['test_bz'] = 1
    if 'scale' not in params['dataset']:
        params['dataset']['scale'] = params['scale']
    if 'scale' not in params['backbone']:
        params['backbone']['scale'] = params['scale']
    return random_params(params)


if __name__ == "__main__":
    submit_jobs(params_for_SR, 'frameworks/superresolution/train_sr_model.py', number_jobs=8, job_directory='.')
