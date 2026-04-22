from src.spatialAdata.loading import load_spatialAdata
from experiments.src.custom_destriping_benchmark.lightweight_funs import (
    save_n_counts_adjusted_counts,
)
import time
import json
import os


def parse_config(original_root, cfg):
    data_path = os.path.join(original_root, cfg.dataset.path_data)
    cell_id_label = cfg.dataset.cell_id_label
    data = load_spatialAdata(data_path)
    data.add_array_coords_to_obs()

    destriping_time_dict = {}

    output_dir = os.path.join(os.getcwd(), "destriped_data")
    start = time.time()
    data.bin_level_normalization(**cfg.model_args, cell_id_label=cell_id_label)
    stop = time.time()
    save_n_counts_adjusted_counts(data, output_dir)

    destriping_time_dict["time_destripe"] = stop - start
    with open("time_dict.json", "w") as json_file:
        json.dump(destriping_time_dict, json_file)
