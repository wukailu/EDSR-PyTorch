import torch

from frameworks.distillation.DEIP import load_model
from frameworks.singlemodel.train_single_model import prepare_params
from model import matmul_on_first_two_dim
from model.layerwise_model import pad_const_channel

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
        'rank_eps': 0.01,  # 0.05, 0.6, 1, 2
        'seed': 0,
        'num_epochs': 1,
        'max_lr': 5e-4,
    }
    params = prepare_params(params)
    model = load_model(params)

    x_test = model.val_dataloader().dataset[0][0]
    x_test = torch.stack([x_test], dim=0)
    f_list, _ = model.teacher_model(x_test, with_feature=True)
    for f in f_list:
        print(f.shape, f.max(), f.min())

    # model.teacher_model = None
    # inference_statics(model, x_test=x_test, batch_size=1)
    # x_test = torch.randn((7, 3, 32, 32))
    xs = x_test.detach()
    xt = x_test.detach()

    with torch.no_grad():
        for layer_s, layer_t, M in zip(model.plain_model, model.teacher_model.sequential_models, model.M_maps):
            conv_s, act_s = layer_s.simplify_layer()
            conv_t, act_t = layer_t.simplify_layer()

            print('teacher_shape, ', conv_t.weight.shape[:2])
            print('student_shape, ', conv_s.weight.shape[:2])

            xs = conv_s(pad_const_channel(xs))
            xt = conv_t(pad_const_channel(xt))

            print(torch.max(torch.abs(matmul_on_first_two_dim(xs, M.transpose(0, 1)) - xt)), torch.max(torch.abs(xt)))

            xs = act_s(xs)
            xt = act_t(xt)

        print(type(model.plain_model[-1]), type(model.teacher_model.sequential_models[-1]))

        xs = model.plain_model[-1](pad_const_channel(xs))
        xt = model.teacher_model.sequential_models[-1](pad_const_channel(xt))
        print(torch.max(torch.abs(xs - xt)), torch.max(torch.abs(xt)))

        print("---------full test--------")

        ps = model(x_test)
        pt = model.teacher_model(x_test)
        print(ps[0], pt[0])
        print(torch.max(torch.abs(ps-pt)), torch.max(torch.abs(pt)))