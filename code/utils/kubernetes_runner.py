import sys
import os

sys.path.append(os.getcwd())

import utils.atlas_backend as backend

namespace = 'wuvin'

# now, we are inside the docker of atlas
if __name__ == "__main__":
    import random
    import string
    job_id = ''.join(random.choice(string.ascii_lowercase) for i in range(12))
    backend.log('job_id = ', job_id)

    # load the params and info
    params = backend.load_parameters()
    job_directory = params['job_directory']
    command = params['command']
    job_params = params['params']
    num_gpus = params['num_gpus']
    backend.log_params(job_params)
    import yaml
    with open('kube_job_parameters.yaml', 'w') as f:
        yaml.dump(job_params, f)

    # modify the kubernetes config
    import yaml
    with open('kubernetes.config.yaml', 'r') as f:
        data = yaml.safe_load(f)
    data['metadata']['name'] = job_id + '-deployment'  # deployment_name
    data['spec']['selector']['matchLabels']['name'] = job_id + '-pod'  # pod name
    data['spec']['template']['metadata']['labels']['name'] = job_id + '-pod'
    data['spec']['template']['spec']['containers'][0]['resources']['limits']['virtaitech.com/gpu'] = num_gpus
    data['spec']['template']['spec']['containers'][0]['resources']['requests']['virtaitech.com/gpu'] = num_gpus
    data['spec']['template']['spec']['containers'][0]['env'][2]['value'] = str(num_gpus)
    data['spec']['replicas'] = 1  # could only creat 1 pod a time

    # start the pod
    from kubernetes import client, config, watch
    config.kube_config.load_kube_config(config_file="kubeconfig.yaml")
    v1 = client.CoreV1Api()
    k8s_apps_v1 = client.AppsV1Api()

    # 部署 Deployment
    resp = k8s_apps_v1.create_namespaced_deployment(body=data, namespace=namespace)
    deployment_name = resp.metadata.name

    import time
    while True:
        resp = k8s_apps_v1.read_namespaced_deployment_status(name=deployment_name, namespace=namespace)
        if resp.status.available_replicas == 1 and resp.status.ready_replicas == 1:
            backend.log('deployment created! available_replicas = ', resp.status.available_replicas)
            break
        time.sleep(1)  # sleep 1 second

    # 列出所有的pod
    pod_name = None
    resp = v1.list_namespaced_pod(namespace=namespace)
    for i in resp.items:
        if i.metadata.name.startswith(job_id):
            pod_name = i.metadata.name
            break
    if pod_name is None:
        raise Exception("Pod not found! job_id = ", job_id)
    backend.log('pod found with pod name = ', pod_name)

    # Calling exec and waiting for response
    from kubernetes.stream import stream

    exec_command = ['/bin/sh', '-c', 'cp -r /data/cache ~/.cache']
    resp = stream(v1.connect_get_namespaced_pod_exec,
                  pod_name,
                  namespace,
                  command=exec_command,
                  stderr=True, stdin=False,
                  stdout=True, tty=False)
    print(resp)

    exec_command = ['/bin/sh', '-c', 'mkdir /job & mkdir /job/job_source']
    resp = stream(v1.connect_get_namespaced_pod_exec,
                  pod_name,
                  namespace,
                  command=exec_command,
                  stderr=True, stdin=False,
                  stdout=True, tty=False)
    print(resp)

    os.system('cp kubernetes.config.yaml ~/.kube/')
    os.system('kubectl cluster-info')
    os.system(f'kubectl cp -r {job_directory} {namespace}/{pod_name}:/job/job_source/')

    # exec_command = ['/bin/sh', '-c', "echo \"This message goes to stderr\" >&2; sleep 5; echo This message is late."]
    exec_command = ['/bin/sh', '-c', 'cd /job/job_source/;' + command]
    resp = stream(v1.connect_get_namespaced_pod_exec,
                  pod_name,
                  namespace,
                  command=exec_command,
                  stderr=True, stdin=True,
                  stdout=True, tty=False,
                  _preload_content=False)
    while resp.is_open():
        resp.update(timeout=100)
        if resp.peek_stdout():
            backend.log("%s" % resp.read_stdout())
        if resp.peek_stderr():
            backend.log("STDERR: %s" % resp.read_stderr())
        time.sleep(1)
    resp.close()

    # TODO: read the artifact list and log list and tensorboard, then move them back
    os.system(f'kubectl cp {namespace}/{pod_name}:/job/job_source/job_info.pkl job_info.pkl')
    import pickle
    with open('job_info.pkl', 'rb') as f:
        job_info = pickle.load(f)
    # job_info = {'params': {}, 'results': {}, 'tensorboard_path': '', 'artifacts': {}}
    if job_info['params'] != {}:
        backend.log_params(job_info['params'])
    if job_info['tensorboard_path'] != '':
        backend.set_tensorboard_logdir(job_info['tensorboard_path'])
        os.system(f'kubectl cp -r {namespace}/{pod_name}:{os.path.join("/job/job_source/", job_info["tensorboard_path"])} ' + job_info['tensorboard_path'])
    for key, path in job_info['artifacts'].items():
        os.system(f'kubectl cp -r {namespace}/{pod_name}:{os.path.join("/job/job_source/", path)} ' + path)
        backend.save_artifact(path, key=key)
    for key, value in job_info['results']:
        backend.log_metric(key, value)
