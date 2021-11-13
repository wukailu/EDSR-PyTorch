import sys, os
sys.path = [os.getcwd()] + sys.path
os.environ['LOCAL_BACKEND'] = '1'

from frameworks.classification.train_single_model import inference_statics
from model import get_classifier
from model.layerwise_model import ConvertibleLayer, pad_const_channel, ConvertibleModel, DenseFeatureFusionSubModel, \
    ConvLayer
import torch

from model.super_resolution_model import RDB_Layerwise

if __name__ == '__main__':
    target_size = (1920, 1080)
    # target_size = (960, 540)

    # params = {
    #     'arch': 'Plain_layerwise_sr',
    #     'n_feats': 90,
    #     'num_modules': 19,
    # }
    # params = {
    #     'arch': 'Plain_layerwise_sr',
    #     'widths': [3, 15, 25, 36, 61, 51, 81, 62, 97, 68, 105, 70, 113, 72, 114, 73, 120, 74, 125, 74, 126, 74, 129, 74, 131, 74, 133, 74, 133, 74, 132, 74, 131, 75, 64],
    #     'scale': 2,
    #     'n_colors': 3,
    # }
    # params = {
    #     'arch': 'Plain_layerwise_sr',
    #     'widths': [3, 15, 26, 41, 72, 58, 93, 67, 107, 72, 113, 74, 119, 75, 119, 75, 127, 75, 127, 75, 132, 75, 134, 75, 132, 75, 135, 75, 133, 74, 132, 74, 129, 75, 64],
    #     'scale': 3,
    #     'n_colors': 3,
    # }
    params = {
        'arch': 'Plain_layerwise_sr',
        'widths': [3, 17, 31, 48, 83, 64, 101, 71, 113, 74, 119, 76, 125, 77, 123, 77, 127, 77, 131, 77, 134, 77, 136, 77, 136, 76, 138, 76, 135, 75, 134, 75, 134, 75, 64],
        'scale': 4,
        'n_colors': 3,
    }
    # params = {
    #     # 'arch': 'srcnn_sr',
    #     # 'arch': 'fsrcnn_sr',
    #     # 'arch': 'VDSR_sr',
    #     # 'arch': 'drrn_sr',
    #     # 'arch': 'memnet_sr',
    #     # 'arch': 'carn_sr',
    #     # 'arch': 'idn_sr',
    #     # 'arch': 'SRFBN_sr',
    #     # 'arch': 'IMDN_sr',
    #     # 'arch': 'EDSR_sr',
    #     'arch': 'EDSR_layerwise_sr',
    #     'simple_tail': True,
    #     'multi_scale': True,
    #     'scale': 2,
    #     'n_colors': 3,
    # }
    print(params['arch'])
    model = get_classifier(params, "DIV2K")
    x_test = torch.randint(0, 255, (2, params['n_colors'], 24, 24)).float()

    model = model.cuda()
    x_test = x_test.cuda()
    with torch.no_grad():
        out = model(x_test)
        assert out.shape == (2, params['n_colors'], 24 * params['scale'], 24 * params['scale'])

    # params = {
    #     'arch': 'resnet20x4_layerwise',
    # }
    # model = get_classifier(params, "cifar100")
    # x_test = torch.randn((1, 3, 32, 32))

    # with torch.no_grad():
    #     f_list, _ = model(x_test, with_feature=True)
    #     for idx, f in enumerate(f_list):
    #         print(idx, ': f.shape', f.shape, 'f.mean', f.mean(), 'f.std', f.std(), 'f.min', f.min(), 'f.max', f.max())
    #
    #     model = ConvertibleModel(model.to_convertible_layers())
    #     f_list_2, _ = model(x_test, with_feature=True)
    #     for idx, f in enumerate(f_list_2):
    #         print(idx, ': f.shape', f.shape, 'f.mean', f.mean(), 'f.std', f.std(), 'f.min', f.min(), 'f.max', f.max())
    #
    #     for f1, f2 in zip(f_list, f_list_2):
    #         f = f1 - f2
    #         print('diff : f.shape', f.shape, 'f.mean', f.mean(), 'f.std', f.std(), 'f.min', f.min(), 'f.max', f.max())

    # ans = x_test.detach()
    # out = x_test.detach()
    # assert isinstance(model, ConvertibleModel)
    # for layer in model.to_convertible_layers()[:-1]:
    #     if isinstance(layer, ConvertibleLayer):
    #         print(type(layer))
    #         conv, act = layer.simplify_layer()
    #         assert isinstance(conv, nn.Conv2d)
    #         with torch.no_grad():
    #             ans = pad_const_channel(ans)
    #             out = pad_const_channel(out)
    #             ans = layer(ans)
    #             out = act(conv(out))
    #             diff_max = (ans - out).abs().max()  # this should be smaller than 1e-5
    #             print("ans shape", ans.shape, "diff_max = ", diff_max, "ans_max", ans.max())
    #             if diff_max > 1e-3:
    #                 print((out-ans).argmax(dim=0)[1], out - ans)
    #                 assert diff_max < 1e-3

    # x_test = torch.randint(0, 255, (1, 16, 2, 2)).float()
    # model = DenseFeatureFusionSubModel([ConvLayer(3, 3, 3, act=nn.ReLU()) for i in range(3)], 3, skip_connection_bias=1000)
    # model = RDB_Layerwise(16, 10, 1)

    # x_test = model(x_test, until=1)
    # x_test = model[1](pad_const_channel(x_test), until=1)[:, :64]
    # model = model[1][1]
    # print(type(model))
    # with torch.no_grad():
        # # out = model(pad_const_channel(x_test))
        # out = model(x_test)
        # out2 = ConvertibleModel.from_convertible_models(model.to_convertible_layers())(x_test)
        # print('diff out out2 = ', (out-out2).abs().max(), 'out_max = ', out.abs().max(), 'out2 max = ', out2.abs().max())
        # print([(out-out2)[:, i].max() for i in range(out.size(1))])

    # print(type(model))
    x_test = torch.randint(0, 255, (params['n_colors'], target_size[0]//params['scale'], target_size[1]//params['scale'])).float()
    inference_statics(model, x_test=x_test, batch_size=1, averaged=False)

### layerwise_rdn
# --------------> Inference_Time(us) = 3.6313236449603683 <-------------
# --------------> Memory(KB) = 8.936342592592593 <-------------
# --------------> flops(K per pixel) = 7602.426666666667 <-------------
# --------------> parameters(KB) = 21814.6279296875 <-------------

### rdn
# --------------> Inference_Time(us) = 2.5558797642588615 <-------------
# --------------> Memory(KB) = 10.248354311342593 <-------------
# --------------> flops(K per pixel) = 7580.069333333333 <-------------
# --------------> parameters(KB) = 21749.1279296875 <-------------