from model import get_classifier
from model.layerwise_model import ConvertibleLayer, pad_const_channel, ConvertibleModel
from torch import nn
import torch

if __name__ == '__main__':
    params = {
        'arch': 'resnet20_layerwise',
    }

    model = get_classifier(params, "cifar100")

    x_test = torch.randn((2, 3, 32, 32))

    ans = x_test
    out = x_test
    assert isinstance(model, ConvertibleModel)
    for layer in model.to_convertible_layers()[:-1]:
        if isinstance(layer, ConvertibleLayer):
            conv, act = layer.simplify_layer()
            assert isinstance(conv, nn.Conv2d)
            with torch.no_grad():
                ans = pad_const_channel(ans)
                out = pad_const_channel(out)
                ans = layer(ans)
                out = act(conv(out))
                diff_max = (ans - out).abs().max()  # this should be smaller than 1e-5
                print("ans shape", ans.shape, "diff_max = ", diff_max, "ans_max", ans.max())

    with torch.no_grad():
        out = model(x_test)
        out2 = ConvertibleModel.from_convertible_models(model.to_convertible_layers())(x_test)
        print('diff out out2 = ', (out-out2).abs().max(), 'out_max = ', out.abs().max(), 'out2 max = ', out2.abs().max())
