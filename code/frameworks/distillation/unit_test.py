import torch

from frameworks.distillation.DEIP import load_model
from frameworks.classification.train_single_model import prepare_params, inference_statics
from model import matmul_on_first_two_dim
from model.layerwise_model import pad_const_channel, ConvertibleModel


def tensor_static(x, name):
    print(name, 'shape', x.shape, 'mean', x.mean(), 'var', x.var(), 'max', x.max(), 'min', x.min())


def super_resolution_test():
    params = {
        'method': 'DEIP_Init',
        'task': 'super-resolution',
        'loss': 'L1',
        'metric': 'psnr255',
        'rgb_range': 255,
        'gpus': 1,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'teacher_pretrain_path': "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_a0131.ckpt",
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
        'ignore_exist': True,
        'save_model': False,
        'project_name': 'deip_SRx4_baseline',
        'add_ori': 0,
        'init_stu_with_teacher': 1,
        'layer_type': 'normal_no_bn',
        'rank_eps': 0.1,  # 0.05, 0.6, 1, 2
        'seed': 0,
        'num_epochs': 1,
        'max_lr': 5e-4,
        'init_with_teacher_param': 1,
    }
    params = prepare_params(params)
    model = load_model(params)

    # ======= speed statistics ===========
    inference_statics(model, batch_size=1)
    # model.plain_model = ConvertibleModel.from_convertible_models(model.plain_model).generate_inference_model()
    # inference_statics(model)

    # ======= teacher student difference ========
    x_test = model.val_dataloader().dataset[0][0]
    x_test = torch.stack([x_test], dim=0)
    # xs = x_test.detach()
    # xt = x_test.detach()
    # cnt=0
    with torch.no_grad():
        # for layer_s, layer_t, M in zip(model.plain_model[:-1], model.teacher_model[:-1], model.bridges[1:]):
        #     conv_s, act_s = layer_s.simplify_layer()
        #     conv_t, act_t = layer_t.simplify_layer()
        #
        #     xs = conv_s(pad_const_channel(xs))
        #     xt = conv_t(pad_const_channel(xt))
        #
        #     xs = act_s(xs)
        #     xt = act_t(xt)
        #
        #     pt = M(pad_const_channel(xs))  # approximation of teacher feature map from student feature map
        #
        #     print('---------layer ', cnt, '--------')
        #     cnt += 1
        #     tensor_static(xs, 'student')
        #     tensor_static(xt, 'teacher')
        #     tensor_static(pt, 'p_teacher')
        #     tensor_static(xt - pt, 'diff')
        #
        # xs = model.plain_model[-1](pad_const_channel(xs))
        # xt = model.teacher_model.sequential_models[-1](pad_const_channel(xt))
        # tensor_static(xs, 'final_student')
        # tensor_static(xt, 'final_teacher')
        # tensor_static(xt - xt, 'diff')

        print("---------full test--------")
        ps = model(x_test, until=-1)
        pt = model.teacher_model(x_test, until=-1)
        tensor_static(ps, 'final_student')
        tensor_static(pt, 'final_teacher')
        tensor_static(ps - pt, 'diff')


def classification_test():
    params = {
        'metric': 'acc',
        'gpus': 1,
        'num_epochs': 300,
        'layer_type': 'normal',
        'weight_decay': 0.0005,
        'max_lr': 0.1,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'SGD',
        'teacher_pretrain_path': '/data/pretrained/lightning_models/layerwise_resnet20x4_cifar100_b8242.ckpt',
        'dataset': {
            'workers': 4,
            'name': 'cifar100',
            'total_batch_size': 256,
            'batch_size': 256
        },
        'seed': 235,
        'method': 'DEIP_Init',
        'init_stu_with_teacher': 0,
        'rank_eps': 0.05,
        'backend': None,
        'learning_rate': 0.1,
        'lr': 0.1
    }
    params = prepare_params(params)
    model = load_model(params)

    x_test = next(iter(model.val_dataloader()))[0]

    with torch.no_grad():
        print("---------full test--------")
        ps = model(x_test)
        pt = model.teacher_model(x_test)
        tensor_static(ps, 'final_student')
        tensor_static(pt, 'final_teacher')
        tensor_static(ps - pt, 'diff')
    # f_list, _ = model.teacher_model(x_test, with_feature=True)
    # for f in f_list:
    #     print(f.shape, f.max(), f.min())
    #
    # xs = x_test.detach()
    # xt = x_test.detach()
    #
    # with torch.no_grad():
    #     for layer_s, layer_t, M in zip(model.plain_model, model.teacher_model.sequential_models, model.M_maps):
    #         conv_s, act_s = layer_s.simplify_layer()
    #         conv_t, act_t = layer_t.simplify_layer()
    #
    #         print('teacher_shape, ', conv_t.weight.shape[:2])
    #         print('student_shape, ', conv_s.weight.shape[:2])
    #
    #         xs = conv_s(pad_const_channel(xs))
    #         xt = conv_t(pad_const_channel(xt))
    #
    #         print(torch.max(torch.abs(matmul_on_first_two_dim(xs, M.transpose(0, 1)) - xt)), torch.max(torch.abs(xt)))
    #
    #         xs = act_s(xs)
    #         xt = act_t(xt)
    #
    #     print(type(model.plain_model[-1]), type(model.teacher_model.sequential_models[-1]))
    #
    #     xs = model.plain_model[-1](pad_const_channel(xs))
    #     xt = model.teacher_model.sequential_models[-1](pad_const_channel(xt))
    #     print(torch.max(torch.abs(xs - xt)), torch.max(torch.abs(xt)))
    #
    #     print("---------full test--------")
    #
    #     ps = model(x_test)
    #     pt = model.teacher_model(x_test)
    #     print(ps[0], pt[0])
    #     print(torch.max(torch.abs(ps - pt)), torch.max(torch.abs(pt)))


if __name__ == '__main__':
    import random
    random.seed(0)
    classification_test()
    # super_resolution_test()
