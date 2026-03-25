"""
Destriping-only experiment for memory profiling.

Loads data and the canonical run's sol, then runs the
destripe_dividing_factors_qm_tot_counts call as produced by
get_all_destripe_calls (cyto: dividing_factors / nucl: qm_tot_counts when
sol.c covers nuclear bins only).  No model fitting.
Run on the cluster and check sacct MaxRSS.
"""
import gc
import json
import os
import pickle
import time
import yaml
from src.spatialAdata.loading import load_spatialAdata
from src.destriping.sol import Sol
from experiments.src.custom_destriping_benchmark.utils import *


def parse_config(original_root, cfg):
    output_dir = os.getcwd()

    # Load benchmark output file for canonical sol / dist paths
    bench_file = os.path.join(original_root, cfg.benchmark_output_file)
    with open(bench_file) as f:
        bench = yaml.safe_load(f)

    data_path = os.path.join(original_root, cfg.dataset.path_data)
    sol_dir = os.path.join(original_root, bench["sol_dir"])
    dist_pkl = os.path.join(original_root, bench["dist_pkl"])

    data = load_spatialAdata(data_path)
    print(data)

    sol = Sol.load(sol_dir)
    with open(dist_pkl, "rb") as f:
        dist_payload = pickle.load(f)
    dist = dist_payload["dist"]
    dist_params = dist_payload["dist_params"]

    cell_id_label = cfg.dataset.cell_id_label

    # MIMIC the get_destripe_call method, but without the apply after copy which is needed only because of benchmark purposes.

    nucl_indices = data.nucl_indices(cell_id_label=cell_id_label)
    method = spatialAdata.destripe_dividing_factors_qm_tot_counts
    cyto_method = spatialAdata.destripe_dividing_factors
    nucl_method = method
    name_method = f"cyto_{cyto_method.__name__}_nucl_{method.__name__}"  # how to get just the last part ?
    args_nucl = get_args_for_method(
        nucl_method, sol, dist=dist, dist_params=dist_params
    )
    args_cyto = get_args_for_method(
        cyto_method, sol, dist=dist, dist_params=dist_params
    )
    args_dict = {
        "method_cyto": cyto_method,
        "method_nucl": nucl_method,
        "nucl_indices": nucl_indices,
        "args_nucl": args_nucl,
        "args_cyto": args_cyto,
    }

    start = time.time()
    destriped_data = data.destripe_combi_nucl_cyto(**args_dict)
    destriping_time = time.time() - start

    with open(os.path.join(output_dir, "time_dict.json"), "w") as f:
        json.dump({"destriping_time": destriping_time}, f)

    del data, sol
    gc.collect()
