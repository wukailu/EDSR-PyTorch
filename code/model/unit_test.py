from frameworks.classification.train_single_model import inference_statics
from model import get_classifier
from model.layerwise_model import ConvertibleLayer, pad_const_channel, ConvertibleModel, DenseFeatureFusionSubModel, \
    ConvLayer
import torch

from model.super_resolution_model import RDB_Layerwise

if __name__ == '__main__':
    params = {
        'arch': 'edsr_layerwise_sr',
        'n_feats': 50,
    }
    model = get_classifier(params, "DIV2K")
    x_test = torch.randint(0, 255, (16, 3, 24, 24)).float()

    # params = {
    #     'arch': 'resnet20x4_layerwise',
    # }
    # model = get_classifier(params, "cifar100")
    # x_test = torch.randn((1, 3, 32, 32))

    with torch.no_grad():
        f_list, _ = model(x_test, with_feature=True)
        for idx, f in enumerate(f_list):
            print(idx, ': f.shape', f.shape, 'f.mean', f.mean(), 'f.std', f.std(), 'f.min', f.min(), 'f.max', f.max())

        model = ConvertibleModel(model.to_convertible_layers())
        f_list_2, _ = model(x_test, with_feature=True)
        for idx, f in enumerate(f_list_2):
            print(idx, ': f.shape', f.shape, 'f.mean', f.mean(), 'f.std', f.std(), 'f.min', f.min(), 'f.max', f.max())

        for f1, f2 in zip(f_list, f_list_2):
            f = f1 - f2
            print('diff : f.shape', f.shape, 'f.mean', f.mean(), 'f.std', f.std(), 'f.min', f.min(), 'f.max', f.max())

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

    print(type(model))
    # x_test = torch.randint(0, 255, (3, 256, 256)).float()
    # inference_statics(model, x_test=x_test, batch_size=1)

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