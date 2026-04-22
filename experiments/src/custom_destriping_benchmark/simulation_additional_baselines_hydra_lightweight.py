from experiments.src.custom_destriping_benchmark.lightweight_funs import (
    save_n_counts_adjusted_counts,
)
import os
import gc
import logging
import pandas as pd
from src.spatialAdata.loading import load_spatialAdata
from .utils import get_all_destripe_calls
from src.destriping.sol import Sol
from src.spatialAdata.loading import load_spatialAdata
import os
import pandas as pd
import hydra
from pathlib import Path
from experiments.src.custom_destriping_benchmark.utils import (
    convert_gen_params_to_qm_params,
    generating_data_distribution,
    make_path_rel_root_lambda,
)


def parse_config(original_root, cfg):
    print(cfg.keys())
    hydra_rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    output_dir = os.path.join(original_root, hydra_rundir)
    additional_baselines_dir = os.path.join(output_dir, "results")

    path_rel_root = make_path_rel_root_lambda(original_root)

    os.makedirs(additional_baselines_dir, exist_ok=True)

    simulation_data_dir = os.path.join(
        original_root, os.path.dirname(cfg.dataset.path_data)
    )
    bin_data_path = os.path.join(simulation_data_dir, "spatial_data")
    cell_id_label = cfg.dataset.cell_id_label

    df_gt_cell_labels = pd.DataFrame(
        columns=["model_name", "fitting_args", "destriping_method"]
    )

    # GT_sol_params
    GT_sol_path = os.path.join(simulation_data_dir, "gt_sol")
    sol = Sol.load(GT_sol_path)
    model = "GT_sol"
    data = load_spatialAdata(bin_data_path)
    # destriping

    ### check if the generating data distribution is poisson or nbinom -> if it was nbinom, on top of poisson, add the nbinom quantile-matching with the correct r###
    distribution, distribution_params = convert_gen_params_to_qm_params(
        *generating_data_distribution(
            os.path.join(original_root, cfg.dataset.path_data)
        )
    )
    if distribution == "nbinom":
        iterator = [("poisson", None), ("nbinom", distribution_params)]

    else:
        iterator = [("poisson", None)]

    output_dir = os.path.join(additional_baselines_dir, "gt_poisson_sol")
    df_gt_poisson_sol_list = []
    for name_dist, distribution_params_ in iterator:
        destriping_calls_ = get_all_destripe_calls(
            sol,
            model,
            data,
            cell_id_label=cell_id_label,
            dist=name_dist,
            dist_params=distribution_params_,
        )
        for name, call in destriping_calls_.items():
            logging.info(name)
            destriped_data = call()
            destriped_data_output_dir = os.path.join(
                output_dir, "destriped_data", name_dist, name
            )
            # lightweight change
            save_n_counts_adjusted_counts(destriped_data, destriped_data_output_dir)

            df_gt_poisson_sol_list.append(
                pd.DataFrame(
                    data={
                        "destriped_data_path": [
                            Path(path_rel_root(destriped_data_output_dir))
                            / "df.parquet"
                        ],
                        "model_name": [f"GT_{name_dist}_sol"],
                        "fitting_args": [{}],
                        "destriping_method": [name],
                        "poisson_sol_path": path_rel_root(GT_sol_path),
                    }
                )
            )
            del destriped_data
            gc.collect()

    ## EXPECTED SDATA WO STRIPES...
    expected_sdata_wo_stripes_path = os.path.join(
        simulation_data_dir, "expected_spatial_data_wo_stripes"
    )
    data = load_spatialAdata(expected_sdata_wo_stripes_path)
    expected_sdata_wo_stripes_output_dir = os.path.join(
        output_dir, "expected_sdata_wo_stripes"
    )
    save_n_counts_adjusted_counts(data, expected_sdata_wo_stripes_output_dir)

    df_gt_poisson_sol_list.append(
        pd.DataFrame(
            data={
                "destriped_data_path": [
                    Path(path_rel_root(expected_sdata_wo_stripes_output_dir))
                    / "df.parquet"
                ],
                "model_name": ["expected_spatial_data_wo_stripes"],
                "fitting_args": [{}],
            }
        )
    )

    df_gt_poisson_sol = pd.concat(df_gt_poisson_sol_list)

    df_baselines_all = pd.concat([df_gt_poisson_sol, df_gt_cell_labels])
    df_baselines_all.reset_index()
    df_baselines_all_path = os.path.join(additional_baselines_dir, "df_baselines.csv")
    df_baselines_all.to_csv(df_baselines_all_path, index=False)
