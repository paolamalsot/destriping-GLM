from pathlib import Path as P
from src.experiments_analysis.analysis import summarize_poisson_sols_with_metrics_from_runs
from src.experiments_analysis.analysis_global_structure import (
    analysis_global_structure,
)
from src.experiments_analysis.analysis import filter_summary_df_qm
import pandas as pd
from pathlib import Path as P
from pathlib import Path as P
from src.experiments_analysis.analysis import (
    summarize_destriped_data_sols_from_runs,
)
from src.utilities.pandas import print_full
from src.utilities.pandas import print_full
from src.experiments_analysis.analysis_dist import distance_fun_dict_same_index
from src.experiments_analysis.analysis import (
    calc_distance_destriped_data_to_gt
)
from pathlib import Path as P
import pandas as pd

def analysis(
    output_dir,
    runs,
    dividing_by_ratio_baselines,
    not_factor_based_baseline,
    supp_baselines_dir,
    synthetic_data = True
):

    P(output_dir).mkdir(parents=True, exist_ok=True)
    global_structure_analysis_folder = P(output_dir) / "global_structure_analysis"
    global_structure_analysis_folder.mkdir(exist_ok=True, parents=True)

    poisson_summary_df = summarize_poisson_sols_with_metrics_from_runs(
        runs,
        dividing_by_ratio_baselines,
        not_factor_based_baseline,
        supp_baselines_dir,
        output_dir=output_dir,
    )

    output_path = P(output_dir) / "poisson_summary_df.pkl"
    poisson_summary_df.to_pickle(output_path)

    output_path = P(output_dir) / "poisson_summary_df.csv"
    poisson_summary_df.to_csv(output_path)

    destriped_summary_df = summarize_destriped_data_sols_from_runs(
        runs,
        dividing_by_ratio_baselines,
        not_factor_based_baseline,
        supp_baselines_dir,
    )

    output_path = P(output_dir) / "destriped_summary_df.pkl"
    destriped_summary_df.to_pickle(output_path)

    output_path = P(output_dir) / "destriped_summary_df.csv"
    destriped_summary_df.to_csv(output_path)

    if synthetic_data:
        print_full(
            poisson_summary_df[
                [
                    "name",
                    "distance_to_gt_poisson_sol_hw_log_euclidian",
                    "converged",
                ]
            ].sort_values(by="distance_to_gt_poisson_sol_hw_log_euclidian")
        )
    else:
        print_full(
            poisson_summary_df[
                [
                    "name",
                    "converged",
                ]
            ]
        )

    output_path = P(output_dir) / "destriped_summary_df.pkl"
    destriped_summary_df = pd.read_pickle(output_path)

    if synthetic_data:
        for gt_destriping_method in ["qm"]:
            destriped_data_df = calc_distance_destriped_data_to_gt(
                destriped_summary_df.copy(),
                distance_fun_dict_=distance_fun_dict_same_index,
                gt_destriping_method=gt_destriping_method,
            )

            output_path = (
                P(output_dir)
                / "distance_destriped_data"
                / f"{gt_destriping_method=}"
                / "destriped_data_df.pkl"
            )
            output_path.parent.mkdir(exist_ok=True, parents=True)
            destriped_data_df.to_pickle(output_path)

            destriped_data_df.to_csv(
                output_path.__str__().replace(".pkl", ".csv"), index=False
            )

    ## global structure analysis
    destriped_summary_df_path = P(output_dir) / "destriped_summary_df.pkl"
    destriped_summary_df = pd.read_pickle(destriped_summary_df_path)
    destriped_summary_df = filter_summary_df_qm(destriped_summary_df)
    destriped_summary_df["fitting_method"] = destriped_summary_df["name"]

    refs_global_structure = [
        "original",
    ]

    if synthetic_data:
        refs_global_structure = refs_global_structure + ["GT_nbinom_sol"]

    destriped_summary_df["name"] = destriped_summary_df["name"].replace(
        {"original__dividing_by_factors": "original"}
    )

    analysis_global_structure(
        destriped_summary_df,
        global_structure_analysis_folder,
        to_plot_global_structure=[],
        refs_global_structure=refs_global_structure,
    )
