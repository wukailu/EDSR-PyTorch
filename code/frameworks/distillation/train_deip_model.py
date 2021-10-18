import sys
import os

sys.path.append(os.getcwd())

from model.layerwise_model import ConvertibleModel
from frameworks.distillation.DEIP import load_model
from frameworks.classification.train_single_model import get_params, prepare_params, train_model, inference_statics
import utils.backend as backend

if __name__ == "__main__":
    params = get_params()
    params = prepare_params(params)
    backend.log(params)
    model = load_model(params)

    if not params['skip_train']:
        model = train_model(model, params, save_name="model_distillation")

    model.plain_model = ConvertibleModel.from_convertible_models(model.plain_model).generate_inference_model()
    if params['inference_statics']:
        if model.params['task'] == 'classification':
            inference_statics(model, batch_size=256)
        else:
            inference_statics(model, batch_size=1)
