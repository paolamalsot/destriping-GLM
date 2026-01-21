from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
import json
import pickle
import pandas as pd
from pathlib import Path as P
from src.experiments_analysis.parse_hydra_output_dirs import (
    get_status_dict, add_poisson_sol_path, get_time_dict, parse_hydra_folder
)
import os

def _normalise_runs_mapping(runs: Mapping[str, str | Path]) -> dict[str, Path]:
    """
    Normalise a mapping {name: path_like} to {name: Path}, dropping
    entries whose paths do not exist locally (e.g. not rsynced yet).
    """
    normalised: dict[str, Path] = {}
    for name, path in runs.items():
        p = Path(path)
        if p.exists():
            normalised[name] = p
        else:
            raise ValueError(f"Path {path} does not exist")
    return normalised


def _load_glm(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def is_glm_run(run_dir):
    glm_path = P(run_dir) / "glm.pkl"
    return os.path.exists(glm_path)


def calculate_glm_specs(run_series):
    run_dir = P(run_series["run_dir"])
    glm_path = run_dir / "glm.pkl"
    dist_path = run_dir / "dist.pkl"
    glm = _load_glm(glm_path)

    with dist_path.open("rb") as handle:
        dist_payload = pickle.load(handle)
    dist_name = dist_payload.get("dist")
    dist_params = dist_payload.get("dist_params")
    alpha = getattr(glm, "alpha_", getattr(glm, "alpha", None))
    theta = getattr(getattr(glm, "family", None), "theta", None)
    n_iter = getattr(glm, "n_iter_", None)
    record = {
        "glm_path": str(glm_path),
        "dist": dist_name,
        "dist_params": dist_params,
        "glm": glm,
        "alpha": alpha,
        "theta": theta,
        "n_iter": n_iter
    }
    return record

def extract_run_supp(run_series):
    dataset_path = run_series["config"]["dataset"]["path_data"]
    cell_id_label = run_series["config"]["dataset"]["cell_id_label"]
    run_dir = P(run_series["run_dir"])
    poisson_sol_path = run_dir / "sol"
    if not(os.path.exists(poisson_sol_path)):
        poisson_sol_path = None

    status_dict_path = run_dir / "status.json"
    time_dict_path = run_dir / "time_dict.json"

    if os.path.exists(time_dict_path):
        time_dict = _load_json(time_dict_path)
    else:
        time_dict = {}

    if (os.path.exists(status_dict_path)):
        status_dict = _load_json(status_dict_path)
    else:
        status_dict = {}

    record: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "cell_id_label": cell_id_label,
        "poisson_sol_path": poisson_sol_path,
        "time_dict": time_dict,
        "status_dict": status_dict,
    }
    return record

def parse_run(name: str, run_dir: Path) -> dict[str, Any]:
    """
    Load a single Hydra run directory into a flat record.

    Expected structure in ``run_dir`` (as produced by
    ``fit_model_destripe_expand_and_aggr_lightweight_glum.parse_config``):
    - opt:``sol/``              : PoissonSol (h, w, c)
    - opt:``status_dict.json``  : dict with model status information
    - opt:``time_dict.json``    : dict with timing information
    - ``.hydra/config.yaml``: full Hydra config (used to recover dataset info)

    Record will contain: overrides, config, run_dir, dataset_path, cell_id_label, poisson_sol_path, time_dict, status_dict
    If Glum run, also: glm_path, dist, dist_params, glm, alpha, theta, n_iter
    """
    run_df = parse_hydra_folder(str(run_dir)) #overrides, config, run_dir
    run_df["fitting_args"] = run_df.config.apply(lambda x: x["model_args"] if "model_args" in x.keys() else {})
    df_time_dict = run_df.apply(get_time_dict, axis=1)  # len(df) =400
    df_status_dict = run_df.apply(
        get_status_dict, axis=1
    )
    run_df = run_df.merge(
        df_time_dict, how="left", left_index=True, right_index=True
    )
    run_df = run_df.merge(
        df_status_dict, how="left", left_index=True, right_index=True
    )
    assert len(run_df) == 1, "df is longer"
    run_series = run_df.iloc[0]
    run_record = extract_run_supp(run_series) #"dataset_path", "cell_id_label", "poisson_sol_path", "time_dict", "status_dict": status_dict,
    if is_glm_run(run_series["run_dir"]):
        supp = calculate_glm_specs(run_series) #"glm_path", "dist", "dist_params", "glm", "alpha", "theta", "n_iter"
    else:
        supp = {}
    return {**run_record, **run_series.to_dict(), **supp, "name": name}


def process_baselines_folders(
    dividing_by_factors_folder_dict, not_factor_based_baseline_folder_dict
):
    list_df_dividing_by_factors = []
    for name, folder in dividing_by_factors_folder_dict.items():
        df_dividing_by_factors = parse_hydra_folder(folder)
        df_dividing_by_factors["poisson_sol_path"] = df_dividing_by_factors.apply(
            add_poisson_sol_path, axis=1
        )
        df_dividing_by_factors["name"] = name
        df_dividing_by_factors["destriping_method"] = "dividing_by_factors"
        list_df_dividing_by_factors.append(df_dividing_by_factors)

    df_dividing_by_factors = pd.concat(list_df_dividing_by_factors, ignore_index=True)

    df_others = []
    for name, folder in not_factor_based_baseline_folder_dict.items():
        df_ = parse_hydra_folder(folder)
        df_["name"] = name
        df_["destriping_method"] = name
        df_others.append(df_)

    df_baselines = pd.concat([df_dividing_by_factors, *df_others])
    df_baselines["destriped_data_path"] = df_baselines["run_dir"].apply(
        lambda x: os.path.join(x, "destriped_data/df.parquet")
    )
    df_baselines["fitting_args"] = df_baselines["overrides"].apply(
        lambda x: {key: x[key] for key in x.keys()}
    )

    df_baselines["dataset_path"] = df_baselines["config"].apply(
        lambda x: x["dataset"]["path_data"]
    )
    df_baselines["cell_id_label"] = df_baselines["config"].apply(
        lambda x: x["dataset"]["cell_id_label"]
    )
    df_baselines.reset_index(drop=True, inplace=True)
    return df_baselines


def collect_runs(runs: Mapping[str, str | Path]) -> pd.DataFrame:
    """
    Collect a small set of runs into a DataFrame.

    Parameters
    ----------
    runs:
        Mapping from a human-readable name (e.g. ``\"regCV_P2_hw_only\"``) to the
        corresponding Hydra run directory produced by the GLUM experiments.

    Returns
    -------
    pd.DataFrame
        Index is the run name. Columns include:
        - ``run_dir``, ``dataset_path``, ``cell_id_label``
        - ``poisson_sol_path``, ``glm_path``, ``dist``, ``dist_params``
        - ``time_dict``, ``status_dict``
        - ``poisson_sol_path`` (PoissonSol), ``glm_path`` (fitted regressor)
        - ``alpha``, ``theta``, ``n_iter`` for glm-runs
    """
    normalised = _normalise_runs_mapping(runs)
    records = [parse_run(name, path) for name, path in normalised.items()]
    df = pd.DataFrame.from_records(records)
    df.set_index("name", inplace=True)
    return df
