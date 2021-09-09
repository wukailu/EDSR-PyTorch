# import frameworks.classification.train_single_model import inference_statics
from model import get_classifier
from model.layerwise_model import ConvertibleLayer, pad_const_channel, ConvertibleModel
from torch import nn
import torch
if __name__ == '__main__':
    params = {
        'arch': 'Plain_layerwise_sr',
        'num_modules': 36,
        'n_feats': 256,
        'add_ori': 1,
    }
    model = get_classifier(params, "DIV2K")
    x_test = torch.randint(0, 255, (2, 3, 24, 24)).float()

    # params = {
    #     'arch': 'resnet20x4_layerwise',
    # }
    # model = get_classifier(params, "cifar100")
    # x_test = torch.randn((1, 3, 32, 32))

    with torch.no_grad():
        f_list, _ = model(x_test, with_feature=True)
        for f in f_list:
            print('f.shape', f.shape, 'f.mean', f.mean(), 'f.var', f.var(), 'f.min', f.min(), 'f.max', f.max())

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
    #             if diff_max > 1e-4:
    #                 print((out-ans).argmax(dim=0)[1], out - ans)
    #                 assert diff_max < 1e-4

    # with torch.no_grad():
    #     out = model(x_test)
    #     out2 = ConvertibleModel.from_convertible_models(model.to_convertible_layers())(x_test)
    #     print('diff out out2 = ', (out-out2).abs().max(), 'out_max = ', out.abs().max(), 'out2 max = ', out2.abs().max())

    # inference_statics(model, x_test=x_test, batch_size=256)

