from src.spatialAdata.loading import load_spatialAdata
from hydra.utils import instantiate
from experiments.src.custom_destriping_benchmark.utils import *
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import time
import json
import logging
import os
import gc
from experiments.src.custom_destriping_benchmark.lightweight_funs import save_n_counts_adjusted_counts
import pickle
import numpy as np
import inspect
from importlib import import_module
from src.destriping.GLUM.glum_loggers import set_level_all_glum_loggers

set_level_all_glum_loggers(logging.DEBUG)

def inspect_signature_from_path(path):
    module_path, class_name = path.rsplit('.', 1)
    module = import_module(module_path)
    class_ = getattr(module, class_name)
    signature = inspect.signature(class_.__init__)
    return signature

def parse_config(original_root, cfg):
    data_path = os.path.join(original_root, cfg.dataset.path_data)
    output_dir = os.getcwd()
    model = cfg.model
    model_cls = model
    model_args = cfg.model_args

    data = load_spatialAdata(data_path)
    print(data)
    # check if the class of model has an attribute timeout, and if yes, set it to the current hydra timeout - 100
    instance = HydraConfig.instance()
    if (
        not (instance.cfg is None) and ("timeout_min" in HydraConfig.get().launcher.keys())
    ):  # debug purposes
        timeout_config = HydraConfig.get().launcher.timeout_min
    else:
        timeout_config = 1000
    if "timeout" in inspect_signature_from_path(model._target_).parameters:
        timeout_sec = (timeout_config - 120) * 60  # convert minutes to seconds
        logging.info(f"Setting timeout for model to {timeout_sec} seconds")
        OmegaConf.set_struct(cfg, False)
        cfg.model_args.timeout=max(timeout_sec, 1)

    if "data_index_selection_path" in cfg.dataset.keys():
        index_selection_path = cfg.dataset.index_selection_path
        index_selection = np.load(index_selection_path, allow_pickle=True)
        train_data = data[index_selection].copy()
    else:
        train_data = data

    model = instantiate(model_cls, data=train_data, **model_args)
    start = time.time()

    model.fit()

    stop = time.time()

    fitting_time = stop - start

    sol_output_dir = os.path.join(output_dir, "sol")
    os.makedirs(sol_output_dir, exist_ok= True)
    sol, dist, dist_params = model.get_sol()
    sol.save(sol_output_dir)

    model_status_dict = model.status_dict

    status_dict_path = os.path.join(output_dir, "status_dict.json")
    with open(status_dict_path, "w") as json_file:
        json.dump(model_status_dict, json_file)

    dist_dict_path = os.path.join(output_dir, "dist.pkl")
    with open(dist_dict_path, "wb") as handle:
        pickle.dump({"dist": dist, "dist_params": dist_params}, handle)

    glm_path = os.path.join(output_dir, "glm.pkl")
    with open(glm_path, "wb") as handle:
        pickle.dump(model.glm_, handle)

    # destriping
    destriping_calls = get_all_destripe_calls(sol, model, data, cell_id_label = cfg.dataset.cell_id_label, dist = dist, dist_params = dist_params)
    destriping_time_dict = {}

    del model
    gc.collect()

    for name, call in destriping_calls.items():
        logging.info(name)
        start = time.time()
        destriped_data = call()
        stop = time.time()
        destriping_time_dict[name] = stop-start
        destriped_data_output_dir = os.path.join(output_dir, "destriped_data", name)
        save_n_counts_adjusted_counts(destriped_data, destriped_data_output_dir)

        del destriped_data
        gc.collect()

    # save time dict
    time_dict = {
        "fitting_time": fitting_time,
        "destripe_time": destriping_time_dict
    }

    with open("time_dict.json", "w") as json_file:
        json.dump(time_dict, json_file)
