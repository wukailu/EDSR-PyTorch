import numpy as np

from utils.tools import get_hparams, pkl_load_artifacts, get_targets, dict_filter


def show_img(results: np.ndarray, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(72, 8))
    plt.matshow(results, vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.title("relation map")
    plt.show()


def merge_results(param_filter):
    return {50000 // get_hparams(t)['dataset_params']['total']: pkl_load_artifacts(t) for t in
            get_targets(param_filter)}


def view_img(results, with_orig=False):
    ret = np.concatenate([results[k] for k in sorted(results.keys())], axis=1)
    if np.min(ret) < 0:
        show_img(ret, vmin=-1)
    else:
        show_img(ret, vmin=0)
    if with_orig:
        show_img(ret)


def show_all(filter_dict, net_name=None):
    view_img(merge_results(dict_filter(filter_dict, net_name)), True)