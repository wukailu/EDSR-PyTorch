import os
import torch
from .fsrcnn import get_fsrcnn_student, get_fsrcnn_teacher
from .plainnet import get_plainnet_student, get_plainnet_teacher

device = None


def get_model(config, model_type):

    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    f = globals().get('get_' + config[model_type+'_model'].name)
    print('get_' + config[model_type+'_model'].name)
    if config[model_type+'_model'].params is None:
        return f()
    else:
        return f(**config[model_type+'_model'].params)



