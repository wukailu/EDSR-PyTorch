import sys
import os

sys.path.append(os.getcwd())

from frameworks.distillation.DEIP import load_model
from frameworks.singlemodel.train_single_model import get_params, prepare_params, train_model, inference_statics
import utils.backend as backend

if __name__ == "__main__":
    params = get_params()
    params = prepare_params(params)
    backend.log(params)
    model = load_model(params)

    if not params['skip_train']:
        model = train_model(model, params, save_name="model_distillation")

    if params['inference_statics']:
        inference_statics(model, batch_size=1)
