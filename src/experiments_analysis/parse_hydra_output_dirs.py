import os.path as path
from os.path import abspath
import pandas as pd
from omegaconf import OmegaConf
import os
import json


def read_hydra_overrides(overrides_path):
    with open(overrides_path, "r") as f:
        overrides = [
            line.lstrip("- ").strip() for line in f.readlines()
        ]  # Remove leading '-' and whitespace
        dict_ = OmegaConf.from_dotlist(overrides)
    return dict_


def get_hydra_overrides(folder):
    hydra_dir = abspath(os.path.join(folder, ".hydra"))
    overrides_path = os.path.join(hydra_dir, "overrides.yaml")
    dict_overrides = read_hydra_overrides(overrides_path)
    return dict_overrides


def get_hydra_config(folder):
    hydra_dir = abspath(os.path.join(folder, ".hydra"))
    config_path = os.path.join(hydra_dir, "config.yaml")
    config = OmegaConf.load(config_path)
    return config


def parse_hydra_sweep_subfolder(run_dir):
    dict_overrides = get_hydra_overrides(run_dir)

    config = get_hydra_config(run_dir)

    return {"overrides": dict_overrides, "config": config, "run_dir": run_dir}


def parse_hydra_sweep_folder(sweep_folder):
    data = []
    for subdir in os.listdir(sweep_folder):
        if not (path.isdir(path.join(sweep_folder, subdir))):
            continue

        if subdir == ".submitit":
            continue

        run_dir = os.path.join(sweep_folder, subdir)
        dict_ = parse_hydra_sweep_subfolder(run_dir)
        dict_supp = {"sweep_folder": sweep_folder, "subdir": subdir}
        dict_ = {**dict_, **dict_supp}
        data.append(dict_)

    return pd.DataFrame.from_records(data)


def parse_hydra_folder(folder):
    # agnostic of whether folder is sweep or single run
    if "multirun.yaml" in os.listdir(folder):
        return parse_hydra_sweep_folder(folder)
    else:
        return pd.DataFrame.from_records([parse_hydra_sweep_subfolder(folder)])


def add_poisson_sol_path(x: pd.Series):
    poisson_sol_path = os.path.join(x.run_dir, "sol")
    return poisson_sol_path


def get_time_dict(x: pd.Series):
    run_dir = x.run_dir
    time_dict_path = os.path.join(run_dir, "time_dict.json")
    with open(time_dict_path) as file:
        dict_ = json.load(file)
    return pd.Series(dict_)


def get_status_dict(x: pd.Series):
    run_dir = x.run_dir
    status_dict_path = os.path.join(run_dir, "status_dict.json")
    with open(status_dict_path) as file:
        dict_ = json.load(file)
    s = pd.Series(dict_)
    s["status_dict"] = dict_
    return s
