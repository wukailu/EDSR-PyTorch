from foundations import load_parameters, log_params
from foundations import set_tensorboard_logdir
from foundations import save_artifact
from foundations import submit

metrics = {}


def log_metric(key, value):
    metrics[key] = value
    import pickle
    with open('metric.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    save_artifact('metric.pkl', key='metric')

    from foundations import log_metric as flog_metric
    flog_metric(key, value)


def log(*args, **kwargs):
    print(*args, **kwargs)


log("using atlas framework")
name = 'atlas_backend'
