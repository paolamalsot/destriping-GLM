import gc
import inspect
import json
import logging
import os
import pickle
import time
from importlib import import_module
from pathlib import Path

import yaml
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

from experiments.src.custom_destriping_benchmark.lightweight_funs import (
    save_n_counts_adjusted_counts,
)
from experiments.src.custom_destriping_benchmark.utils import get_all_destripe_calls
from src.destriping.GLUM.glum_loggers import set_level_all_glum_loggers
from src.spatialAdata.spatialAdata import spatialAdata
from src.spatialAdata.loading import load_spatialAdata

set_level_all_glum_loggers(logging.DEBUG)


def inspect_signature_from_path(path):
    module_path, class_name = path.rsplit(".", 1)
    module = import_module(module_path)
    class_ = getattr(module, class_name)
    return inspect.signature(class_.__init__)

def collapse_label(data: spatialAdata, cell_id_label):
    # collapse all nuclei indices to the first nucleic index
    nucl_mask = data.nucl_mask(cell_id_label)
    first_index = data.obs.loc[nucl_mask, cell_id_label].iloc[0]
    data.obs.loc[nucl_mask, cell_id_label] = first_index
    return data

def parse_config(original_root, cfg):
    output_dir = os.getcwd()
    model_cls = cfg.model
    model_args = cfg.model_args
    cell_id_label = cfg.dataset.cell_id_label

    path_data = Path(original_root) / cfg["dataset"]["path_data"]
    data = load_spatialAdata(str(path_data))
    data = collapse_label(data, cell_id_label)

    instance = HydraConfig.instance()
    if not (instance.cfg is None) and (
        "timeout_min" in HydraConfig.get().launcher.keys()
    ):
        timeout_config = HydraConfig.get().launcher.timeout_min
    else:
        timeout_config = 1000
    if "timeout" in inspect_signature_from_path(model_cls._target_).parameters:
        timeout_sec = (timeout_config - 120) * 60
        logging.info(f"Setting timeout for model to {timeout_sec} seconds")
        OmegaConf.set_struct(cfg, False)
        cfg.model_args.timeout = max(timeout_sec, 1)

    model = instantiate(model_cls, data=data, **model_args)
    start = time.time()
    model.fit()
    stop = time.time()
    fitting_time = stop - start

    sol_output_dir = os.path.join(output_dir, "sol")
    os.makedirs(sol_output_dir, exist_ok=True)
    sol, dist, dist_params = model.get_sol()
    sol.save(sol_output_dir)

    with open(os.path.join(output_dir, "status_dict.json"), "w") as f:
        json.dump(model.status_dict, f)

    with open(os.path.join(output_dir, "dist.pkl"), "wb") as f:
        pickle.dump({"dist": dist, "dist_params": dist_params}, f)

    with open(os.path.join(output_dir, "glm.pkl"), "wb") as f:
        pickle.dump(model.glm_, f)

    destriping_calls = get_all_destripe_calls(
        sol,
        model,
        data,
        cell_id_label=cell_id_label,
        dist=dist,
        dist_params=dist_params,
    )
    destriping_time_dict = {}

    del model
    gc.collect()

    for name, call in destriping_calls.items():
        logging.info(name)
        start = time.time()
        destriped_data = call()
        stop = time.time()
        destriping_time_dict[name] = stop - start
        destriped_data_output_dir = os.path.join(output_dir, "destriped_data", name)
        save_n_counts_adjusted_counts(destriped_data, destriped_data_output_dir)
        del destriped_data
        gc.collect()

    time_dict = {"fitting_time": fitting_time, "destripe_time": destriping_time_dict}
    with open("time_dict.json", "w") as f:
        json.dump(time_dict, f)
