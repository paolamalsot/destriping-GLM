"""
Fitting-only experiment for memory profiling.

Loads data, fits GlumWrapper with max_iter=1 / fit_theta_max_iter=1.  No destriping.  Run on the cluster and check sacct MaxRSS.
"""
import gc
import json
import logging
import os
import pickle
import time

from hydra.utils import instantiate
import pandas as pd
from pathlib import Path as P 
from src.spatialAdata.loading import load_spatialAdata
from src.destriping.GLUM.glum_loggers import set_level_all_glum_loggers

set_level_all_glum_loggers(logging.DEBUG)


def parse_config(original_root, cfg):

    df = pd.read_pickle(P(original_root) / cfg.df_path)
    print(df)

    model = instantiate(cfg.model, data_df=df, **cfg.model_args)

#    start = time.time()
    model.fit()
#    fitting_time = time.time() - start

    # sol_output_dir = os.path.join(output_dir, "sol")
    # os.makedirs(sol_output_dir, exist_ok=True)
    # sol, dist, dist_params = model.get_sol()
    # sol.save(sol_output_dir)

    # with open(os.path.join(output_dir, "dist.pkl"), "wb") as f:
    #     pickle.dump({"dist": dist, "dist_params": dist_params}, f)

    # with open(os.path.join(output_dir, "status_dict.json"), "w") as f:
    #     json.dump(model.status_dict, f)

    # with open(os.path.join(output_dir, "time_dict.json"), "w") as f:
    #     json.dump({"fitting_time": fitting_time}, f)

    # del model, sol
    # gc.collect()
