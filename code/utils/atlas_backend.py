from foundations import log_metric
from foundations import load_parameters, log_params
from foundations import set_tensorboard_logdir
from foundations import save_artifact
from foundations import submit


def log(info):
    print(info)


log("using atlas framework")
name = 'atlas_backend'
