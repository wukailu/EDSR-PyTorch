from datasets.utils import Normalize
from model import get_classifier
import torch
import time

if __name__ == '__main__':
    params = {
        'arch': 'Plane_sr',
        'scale': 3,
        'nf': 50
    }

    model = get_classifier(params, "div2k")
    model.cuda().eval()
    x_test = torch.randint(0, 256, (16, 3, 24, 24)).float().cuda()
    with torch.no_grad():
        outs = model(x_test)
        print(outs.shape)

    # backbones = [
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': False,
    #     #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 64},
    #     # {'arch': 'rfdn_sr'},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
    #     #  'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 1, 'nf': 64},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
    #     #  'use_act': False, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 1, 'nf': 128},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
    #     #  'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 4, 'num_modules': 1, 'nf': 64},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'bn', 'block_skip': True, 'add_ori': False,
    #     #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 1, 'num_modules': 6, 'nf': 128},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
    #     #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 2, 'nf': 128},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': True, 'add_ori': True,
    #     #  'use_act': False, 'add_fea': False, 'use_esa': False, 'sub_blocks': 1, 'num_modules': 3, 'nf': 128},
    #     # {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'spade', 'block_skip': False, 'add_ori': True,
    #     #  'use_act': True, 'add_fea': False, 'use_esa': True, 'sub_blocks': 2, 'num_modules': 4, 'nf': 50},
    #     {'arch': 'inn_sr', 'version': 'new_spade_act', 'norm_type': 'bn', 'block_skip': False, 'add_ori': False,
    #      'use_act': True, 'add_fea': False, 'use_esa': False, 'sub_blocks': 2, 'num_modules': 4, 'nf': 50},
    # ]

    # backbones = [
    #     {'arch': 'Plane_sr', 'nf': 128, 'num_modules': 3, 'norm_type': 'spade', 'conv_in_block': 2, 'use_act': True,
    #      'norm_before_relu': True, 'use_esa': True, 'use_spade': False, 'large_ori': True, 'scale': 4}
    # ]
    # for model_params in backbones:
    #     model = get_classifier(model_params, "div2k")
    #     torch.cuda.empty_cache()
    #     torch.cuda.reset_max_memory_allocated()
    #     x_test = torch.randint(0, 256, (16, 3, 96, 96)).float().cuda()
    #     model.cuda().eval()
    #     with torch.no_grad():
    #         total_time = 0
    #         for i in range(10):
    #             outs = model(x_test)
    #             print(outs.shape)
    #         for i in range(10):
    #             torch.cuda.synchronize()
    #             start_time = time.process_time()
    #             outs = model(x_test)
    #             total_time += time.process_time() - start_time
    #         used_memory = torch.cuda.max_memory_allocated()
    #         print('Inference_Time(ms)', float(total_time / 10 * 1000))
    #         print('Memory(MB)', int(used_memory / 1024 / 1024))

    # from torchvision.models import vgg16_bn
    #
    # vgg = vgg16_bn(pretrained=True).cuda()
    # x = torch.randint(0, 256, (2, 3, 96, 96)).float().cuda()
    # with torch.no_grad():
    #     norm = Normalize(mean=(0.485*255, 0.456*255, 0.406*255), std=(0.229*255, 0.224*255, 0.225*255)).cuda()
    #     features = [m for m in vgg16_bn(pretrained=True).features]
    #     print(features)
    #     x = norm(x)
    #     print(x.mean(), x.std())
    #     features = torch.nn.Sequential(*features[:12]).cuda()
    #     x = features(x)
    #     print(x.shape)
