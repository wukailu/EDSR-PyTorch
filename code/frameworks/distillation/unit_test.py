from frameworks.singlemodel.train_single_model import prepare_params, inference_statics
from frameworks.distillation.DEIP import load_model
from model import get_classifier, matmul_on_first_two_dim
import torch

if __name__ == '__main__':
    params = {
        'task': 'super-resolution',
        'loss': 'L1',
        'metric': 'psnr255',
        'rgb_range': 255,
        'gpus': 1,
        'weight_decay': 0,
        'lr_scheduler': 'OneCycLR',
        'optimizer': 'Adam',
        'teacher_pretrain_path': "/data/pretrained/lightning_models/layerwise_edsrx4_div2k_fc971.ckpt",
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
        'init_stu_with_teacher': 0,
        'layer_type': 'normal_no_bn',
        'rank_eps': 0.4,  # 0.05, 0.6, 1, 2
        'seed': 0,
        'num_epochs': 1,
        'max_lr': 5e-4,
    }
    params = prepare_params(params)
    model = load_model(params)

    x_test = model.val_dataloader().dataset[0][0]
    x = torch.stack([x_test], dim=0)
    f_list, _ = model.teacher_model(x, with_feature=True)
    for f in f_list:
        print(f.shape, f.max(), f.min())

    # model.teacher_model = None
    inference_statics(model, x_test=x_test, batch_size=1)
    # x_test = torch.randn((7, 3, 32, 32))
    # xs = x_test.detach()
    # xt = x_test.detach()
    #
    # with torch.no_grad():
    #     # for layer_s, layer_t, M in zip(model.plain_model, model.teacher_model.sequential_models, model.M_maps):
    #     #     conv_s = layer_s[0]
    #     #     conv_t = layer_t.simplify_layer()[0]
    #     #
    #     #     print('teacher_shape, ', conv_t.weight.shape[:2])
    #     #     print('student_shape, ', conv_s.weight.shape[:2])
    #     #     # conv_s.bias.data = torch.zeros_like(conv_s.bias.data)
    #     #     # conv_t.bias.data = torch.zeros_like(conv_t.bias.data)
    #     #
    #     #     xs = conv_s(xs)
    #     #     xt = conv_t(xt)
    #     #
    #     #     print(torch.max(torch.abs(matmul_on_first_two_dim(xs, M.transpose(0, 1)) - xt)), torch.max(torch.abs(xt)))
    #     #
    #     #     # xs = nn.ReLU()(xs)
    #     #     # xt = nn.ReLU()(xt)
    #     #
    #     # print(type(model.plain_model[-1]), type(model.teacher_model.sequential_models[-1]))
    #     #
    #     # xs = model.plain_model[-1](xs)
    #     # xt = model.teacher_model.sequential_models[-1](xt)
    #     # print(torch.max(torch.abs(xs - xt)), torch.max(torch.abs(xt)))
    #
    #     # ps = model(x_test)
    #     # pt = model.teacher_model(x_test)
    #     # print(ps[0], pt[0])
    #     # print(torch.max(torch.abs(ps-pt)), torch.max(torch.abs(pt)))
    #
    #     fs, _ = model(x_test, with_feature=True)
    #     ft, _ = model.teacher_model(x_test, with_feature=True)