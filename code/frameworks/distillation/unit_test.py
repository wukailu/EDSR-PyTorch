from frameworks.singlemodel.train_single_model import prepare_params, inference_statics
from frameworks.distillation.DEIP import load_model
from model import matmul_on_first_two_dim, get_classifier
import torch

if __name__ == '__main__':
    params = {
        'init_stu_with_teacher': True,
        # 'layer_type': 'repvgg',
        'gpus': 1,
        'num_epochs': 300,
        'track_grad_norm': True,
        'rank_eps': 1,
        'weight_decay': 5e-4,
        'max_lr': 5e-2,
        'optimizer': 'SGD',
        'teacher_pretrain_path': "/data/pretrained/lightning_models/layerwise_resnet20_cifar100_58603.ckpt",
        "dataset": {'name': "cifar100", 'total_batch_size': 256},
        "seed": 0,
    }
    params = prepare_params(params)
    model = load_model(params)

    inference_statics(model, x_test=torch.randn((3, 32, 32)))
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