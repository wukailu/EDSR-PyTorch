job_info = {'params': {}, 'results': {}, 'tensorboard_path': '', 'artifacts': {}}
name = 'kube_backend'


def save_job_info():
    import pickle
    with open('/job/job_source/job_info.pkl', 'wb') as f:
        pickle.dump(job_info, f)


def log_metric(key, value):
    job_info['results'][key] = value
    save_job_info()


def log_param(key, value):
    log_params({key: value})


def log_params(parameters):
    job_info['params'] = parameters
    save_job_info()


def set_tensorboard_logdir(path):
    job_info['tensorboard_path'] = path
    save_job_info()


def save_artifact(filepath: str, key=None):
    import random
    if key is None:
        key = str(random.randint(0, 9999)) + "_" + filepath.split('/')[-1].split('.')[0]
    job_info['artifacts'][key] = filepath
    save_job_info()


def log(*info):
    print(*info)


def submit(job_directory, command, params, num_gpus, **kwargs):
    runner_params = {
        'job_directory': job_directory,
        'command': 'python -W ignore ' + command,
        'params': params,
        'num_gpus': num_gpus
    }

    from utils.atlas_backend import submit as atlas_submit
    atlas_submit(job_directory='.', command='utils/kubernetes_runner.py', params=runner_params, num_gpus=0, **kwargs)


def load_parameters(log_parameters=True):
    import yaml
    with open('kube_job_parameters.yaml', 'r') as f:
        param = yaml.safe_load(f)
    return param
