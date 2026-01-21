from src.spatialAdata.loading import load_spatialAdata
from experiments.src.custom_destriping_benchmark.lightweight_funs import save_n_counts_adjusted_counts
from src.destriping.init_params import quantile_init_from_sdata, ones_init_from_sdata, ratio_init_from_sdata
from src.destriping.utils.make_sol import make_lightweight_sol
import time
import json
import os


def parse_config(original_root, cfg):

    data_path = os.path.join(original_root, cfg.dataset.path_data)
    cell_id_label = cfg.dataset.cell_id_label
    data = load_spatialAdata(data_path)
    data.add_array_coords_to_obs()

    destriping_time_dict = {}

    main_output_dir = os.getcwd()

    start = time.time()
    if cfg.factors == "ones":
        h, w = ones_init_from_sdata(data)
    elif cfg.factors == "quantiles":
        h, w = quantile_init_from_sdata(data, quant_init = cfg.quant_init)
    elif cfg.factors == "quantiles_nucl":
        h, w = quantile_init_from_sdata(data, quant_init = cfg.quant_init, nucl_only = True, cell_id_label = cell_id_label)
    elif cfg.factors == "median_ratio":
        h, w = ratio_init_from_sdata(data, quantile = cfg.quant_init, cell_id_label = cell_id_label)
    else:
        raise ValueError("Wrong value for config factors")

    sol = make_lightweight_sol(h, w, data)

    sol_dir = os.path.join(main_output_dir, "sol")
    os.makedirs(sol_dir, exist_ok= True)
    sol.save(sol_dir)

    start = time.time()
    data = data.destripe_dividing_factors(h, w)
    stop = time.time()
    destriped_data_output_dir = os.path.join(main_output_dir, "destriped_data")
    os.makedirs(destriped_data_output_dir, exist_ok=True)
    # lightweight change
    save_n_counts_adjusted_counts(data, destriped_data_output_dir)

    destriping_time_dict["time_destripe"] = stop-start
    with open("time_dict.json", "w") as json_file:
        json.dump(destriping_time_dict, json_file)
