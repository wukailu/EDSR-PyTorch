from model import get_classifier
import torch

from model.super_resolution_model.inn_model import RFDB, ESA, Rep_RFDB

if __name__ == '__main__':
    # x_test = torch.randn(2, 3, 32, 32)
    # net = get_classifier("resnet18", "cifar100")
    # feats, logit = net(x_test, with_feature=True)
    #
    # for f in feats:
    #     print(f.shape, f.min().item())
    # print(logit.shape)
    # print(type(net))

    # 43.6 ms spade_act_shallower_noesa
    # 51.5 ms spade_act_shallower
    # 90.4 ms spade_act_shallow(2 layers)
    # 143.3 ms rfdn


    x_test = torch.randint(0, 256, (64, 3, 48, 48)).float()
    y_test = torch.randint(0, 256, (16, 3, 192, 192)).float()
    net = get_classifier("inn_sr", "div2k")
    with torch.no_grad():
        for i in range(10):
            torch.cuda.synchronize()
            outs = net(x_test)
        RFDB.total_time = 0
        Rep_RFDB.total_time = 0
        ESA.total_time = 0
        for i in range(10):
            torch.cuda.synchronize()
            outs = net(x_test)
        print(outs.shape)
        print(RFDB.total_time)
        print(Rep_RFDB.total_time)
        print(ESA.total_time)
