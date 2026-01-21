from __future__ import annotations
from warnings import warn
import numpy as np
from src.destriping.GLUM.get_metrics_dist import get_metrics_dist
from src.experiments_analysis.analysis_utils import (
    build_df_from_sdata,
    get_cell_id_label,
    get_dataset_path,
    get_destriped_tot_counts,
    get_destriped_tot_counts_from_df,
    get_df_destriped_data_from_row,
    load_gt_sol,
    get_index_gt_destriped,
)
from src.experiments_analysis.glum_collect_runs import (
    collect_runs,
    process_baselines_folders,
)
from src.destriping.sol import Sol
from src.experiments_analysis.parse_hydra_output_dirs import parse_hydra_sweep_subfolder
from src.spatialAdata.loading import load_spatialAdata
import src.utilities.cv2_utils as cv2_utils
from src.utilities.utils import warn_with_prefix
import pandas as pd
from pathlib import Path, Path as P
from typing import Mapping
import ast
import os


def load_df(path):
    df = pd.read_csv(path, low_memory=False)
    df["fitting_args"] = df["fitting_args"].apply(ast.literal_eval)
    return df


def load_supp_baselines(dir_):
    # dir_ is for example results/benchmark_destriping_model/simulation/additional_baselines/2025-05-08/10-59-06/0__+group=['dataset=simulation_1']
    df_supp_baselines = load_df(Path(dir_) / "results" / "df_baselines.csv")
    series_config = pd.Series(parse_hydra_sweep_subfolder(dir_))
    # The following won't work because the series doesn't only contain scalar values
    # df_supp_baselines = df_supp_baselines.assign(**series_config)
    S_df = pd.DataFrame([series_config] * len(df_supp_baselines)).reset_index(drop=True)
    df_supp_baselines = pd.concat(
        [df_supp_baselines.reset_index(drop=True), S_df], axis=1
    )
    df_supp_baselines["dataset_path"] = df_supp_baselines["config"].apply(
        lambda x: x["dataset"]["path_data"]
    )
    df_supp_baselines["cell_id_label"] = df_supp_baselines["config"].apply(
        lambda x: x["dataset"]["cell_id_label"]
    )
    return df_supp_baselines


def compute_sol_metrics(row: pd.Series, df, gt_sol) -> pd.Series:
    """
    Compute metrics for a solution.

    Metrics include:
    - distances to the ground-truth Poisson solution (if available):
      ``distance_to_gt_poisson_sol_hw_log_euclidian``,
      ``distance_to_gt_poisson_sol_f_log_euclidian``, plus the full set returned
      by ``get_metrics_dist``;
    - global structure statistics from ``global_structure_statistics`` such as
      ``hw_acf_energy`` and ``hw_abs_lag1_autocorrelation``;
    """

    poisson_sol_path = row["poisson_sol_path"]
    if not (pd.isna(poisson_sol_path)):
        fitted_sol = Sol.load(poisson_sol_path)

        metrics = {}
        if gt_sol is not None:
            dist_metrics = get_metrics_dist(fitted_sol, gt_sol, df)
            metrics.update(dist_metrics)

        return pd.Series(metrics)

    else:
        return pd.Series({})


def summarize_poisson_sols_with_metrics_from_runs(
    runs: Mapping[str, str | Path],
    dividing_by_factors_folders_dict,
    other_baselines_folders_dict,
    supp_baselines_dir=None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Summarise poisson solutions from a collection of runs & baselines

    Parameters
    ----------
    runs:
        Mapping ``{name: dir}`` where ``dir`` is a Hydra run directory or a sweep run directory
        produced by the GLUM experiments (e.g. from the penalty search).
    dividing_by_factors_folders_dict:
        Mapping ``{name: dir}`` where ``dir`` is a Hydra run directory or a sweep run directory
        produced by the dividing_by_factors baseline
    other_baselines_folders_dict:
        Mapping ``{name: dir}`` where ``dir`` is a Hydra run directory or a sweep run directory
        produced by the other baseline which are not factor/sol based !
    output_dir:
        Optional directory where plots should be written. When ``None``, nothing is written to disk and the
        DataFrame is only returned.
    supp_baselines_dir: directory from supplementary_baselines_runs (often with ground-truth etc...)

    Returns
    -------
    pd.DataFrame
        One row per run name. Columns include:
        - basic metadata: ``dataset_path``, ``run_dir``;
        - distances to the ground-truth Poisson solution (when available),
          including ``distance_to_gt_poisson_sol_hw_log_euclidian`` and
          ``distance_to_gt_poisson_sol_f_log_euclidian``;
        - global structure statistics such as ``hw_acf_energy`` and
          ``hw_abs_lag1_autocorrelation``;
        - optimisation diagnostics: ``alpha``, ``theta``, ``n_iter``, and
          ``fitting_time``.
    """

    df_runs = collect_runs(runs)
    df_runs.reset_index("name", inplace=True)
    df_other = process_baselines_folders(
        dividing_by_factors_folders_dict, other_baselines_folders_dict
    )

    to_concat = [df_runs, df_other]
    if supp_baselines_dir is not None:
        df_supp_baselines = load_supp_baselines(supp_baselines_dir)
        df_supp_baselines.rename(columns={"model_name": "name"}, inplace=True)
        df_supp_baselines.drop_duplicates(subset="poisson_sol_path", inplace=True)
        to_concat.append(df_supp_baselines)

    df_runs = pd.concat(to_concat, ignore_index=True, axis=0)
    df_runs.reset_index()

    dataset_path = get_dataset_path(df_runs)
    cell_id_label = get_cell_id_label(df_runs)
    dataset_df = build_df_from_sdata(dataset_path, cell_id_label)
    dataset_root = P(dataset_path).parent
    gt_sol = load_gt_sol(dataset_root)

    def fun_(row):
        with warn_with_prefix(f"computing sol metrics for {row['name']}: "):
            return compute_sol_metrics(row, dataset_df, gt_sol)

    records_df = df_runs.apply(fun_, axis=1)

    summary_df = pd.concat([df_runs, records_df], axis=1)

    if (output_dir is not None) and (gt_sol is not None):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "simulation_summary.csv"
        summary_df.to_csv(summary_path)

    return summary_df


def check_nan_inf_value(row, n_counts_adjusted=True):
    destriped_data = get_destriped_tot_counts(row, n_counts_adjusted)
    return np.any(np.isnan(destriped_data) | np.isinf(destriped_data))


def check_nan_inf_destriped_data(destriped_summary_df):
    for n_counts_adjusted in [True, False]:
        colname = f"nan_inf_destriped_data__{n_counts_adjusted=}"
        destriped_summary_df[colname] = destriped_summary_df.apply(
            check_nan_inf_value, n_counts_adjusted=n_counts_adjusted, axis=1
        )
    return destriped_summary_df


def add_processed_data_paths_lightweight(x: pd.Series):
    data_dicts = get_processed_data_dicts_lightweight(
        os.path.join(x.run_dir, "destriped_data")
    )
    res = pd.DataFrame.from_records(data_dicts)
    res.index = [x.run_dir] * len(res.index)
    return res


def summarize_destriped_data_sols_from_runs(
    runs: Mapping[str, str | Path],
    dividing_by_factors_folders_dict,
    other_baselines_folders_dict,
    supp_baselines_dir=None,
) -> pd.DataFrame:
    """
    High-level entry point: summarise a small collection of GLUM runs on
    simulated data.

    Parameters
    ----------
    runs:
        Mapping ``{name: dir}`` where ``dir`` is a Hydra run directory or a sweep run directory
        produced by the GLUM experiments (e.g. from the penalty search) or by the other experiments (e.g. baselines, bin2cell, etc...).
    output_dir:
        Optional directory where the summary DataFrame and diagnostic plots
        should be written. When ``None``, nothing is written to disk and the
        DataFrame is only returned.

    Returns
    -------
    pd.DataFrame
        One row per run name. Columns include:
        - basic metadata: ``dataset_name``, ``dataset_path``, ``run_dir``;
        - distances to the ground-truth Poisson solution (when available),
          including ``distance_to_gt_poisson_sol_hw_log_euclidian`` and
          ``distance_to_gt_poisson_sol_f_log_euclidian``;
        - global structure statistics such as ``hw_acf_energy`` and
          ``hw_abs_lag1_autocorrelation``;
        - optimisation diagnostics: ``alpha``, ``theta``, ``n_iter``, and
          ``fitting_time``.
    """
    df = collect_runs(runs)
    df.reset_index("name", inplace=True)
    df_other = process_baselines_folders(
        dividing_by_factors_folders_dict, other_baselines_folders_dict
    )
    temp = pd.concat(df.apply(add_processed_data_paths_lightweight, axis=1).values)
    df = df.merge(right=temp, left_on="run_dir", right_index=True, how="outer")

    to_concat = [df, df_other]
    if supp_baselines_dir is not None:
        df_supp_baselines = load_supp_baselines(supp_baselines_dir)
        df_supp_baselines.rename(columns={"model_name": "name"}, inplace=True)
        to_concat.append(df_supp_baselines)

    df_final = pd.concat(to_concat, axis=0)
    df_final.reset_index(inplace=True, drop=True)

    # check presence of NaN or inf in destriped data
    df_final = check_nan_inf_destriped_data(df_final)
    if (
        df_final["nan_inf_destriped_data__n_counts_adjusted=True"].any()
        or df_final["nan_inf_destriped_data__n_counts_adjusted=True"].any()
    ):
        list_nan = df_final.query(
            "nan_inf_destriped_data__n_counts_adjusted=True or nan_inf_destriped_data__n_counts_adjusted=True"
        )["name"].tolist()
        warn(f"NaN inf in the following destriped data: {list_nan}")

    return df_final


def get_processed_data_dicts_lightweight(dir_):
    list_dicts = []
    for subdir in os.listdir(dir_):  # remove self...
        if subdir == ".DS_Store":
            continue

        dict_ = {}
        dict_["destriping_method"] = subdir
        dict_["destriped_data_path"] = os.path.join(dir_, subdir, "df.parquet")
        assert os.path.exists(dict_["destriped_data_path"]), print(
            f"{dict_['destriped_data_path']} does not exist"
        )
        list_dicts.append(dict_)

    return list_dicts


def filter_summary_df_qm(destriped_summary_df):
    def select_method(destripe_method_df):
        for method in destripe_method_df["destriping_method"].tolist():
            if pd.isna(method):
                continue
            if ("qm" in method) and not (
                "sqm" in method
            ):  # for compatibility with my previous results
                return destripe_method_df.query("destriping_method == @method").iloc[0]
        assert len(destripe_method_df) == 1
        return destripe_method_df.iloc[0]

    destriped_summary_df = (
        destriped_summary_df.groupby("name").apply(select_method).reset_index(drop=True)
    )
    return destriped_summary_df


def destriped_data_plots(global_structure_dir):
    matrices_output_folder = P(global_structure_dir) / "destriping_matrices"
    destriped_data_plots_dir = P(global_structure_dir) / "destriped_data_plots"
    df_matrices = pd.read_csv(matrices_output_folder / "df_results_path_matrices.csv")

    P(destriped_data_plots_dir).mkdir(exist_ok=True, parents=True)

    for name in df_matrices["name"].tolist():
        matrix_path = df_matrices.query(f"name == '{name}'")[
            "path_destriped_n_counts_matrix"
        ].item()
        matrix = np.load(matrix_path)
        path_fig = P(destriped_data_plots_dir) / f"{name}.tif"
        if np.any(np.isnan(matrix)):
            matrix_plot = matrix.copy()
            matrix_plot[np.isnan(matrix)] = 0
        else:
            matrix_plot = matrix
        cv2_utils.save_tif_img_with_colorbar(
            matrix_plot, path_fig.__str__(), log1p=False
        )


def calc_distance_destriped_data_to_gt(
    df, distance_fun_dict_, gt_destriping_method="qm"
):
    """Calculate distance of destriped data to ground truth for each dataset."""

    df = df.rename(columns={"name": "model_name"})

    original_data = load_spatialAdata(df["dataset_path"].unique()[0])
    original_data.n_counts
    nuclear_indices = original_data.nucl_indices()
    for dataset_path, df_dataset in df.groupby("dataset_path"):
        df_results = df_dataset.copy()

        df_dataset_ = df_dataset.loc[~pd.isna(df_dataset["destriped_data_path"])]
        if df_dataset_.empty:
            continue

        index_gt = get_index_gt_destriped(
            dataset_path, df_dataset_, gt_destriping_method
        )
        gt_poisson_sol_line = df_dataset_.loc[index_gt]
        gt_poisson_data_tot_counts = get_destriped_tot_counts(gt_poisson_sol_line)
        gt_poisson_data_tot_counts_nucl = gt_poisson_data_tot_counts.loc[
            nuclear_indices
        ]

        def distance_to_gt(row):
            df_ = get_df_destriped_data_from_row(row)
            result_dict = {}
            for take_n_counts_adjusted in [True, False]:
                destriped_data_tot_counts = get_destriped_tot_counts_from_df(
                    df_, take_n_counts_adjusted=take_n_counts_adjusted
                )

                colname = f"NA_in_take_n_counts_adjusted={take_n_counts_adjusted}"
                na = destriped_data_tot_counts.isna()
                result_dict[colname] = na.any()

                colname = f"INF_in_take_n_counts_adjusted={take_n_counts_adjusted}"
                inf = np.isinf(destriped_data_tot_counts.values)
                result_dict[colname] = np.any(inf)

                # filling the na and inf with the original
                pd.testing.assert_index_equal(
                    destriped_data_tot_counts.index, original_data.obs.index
                )
                to_repl = inf | na.values
                destriped_data_tot_counts.loc[to_repl] = original_data.obs.loc[
                    to_repl, "n_counts"
                ]

                for nucleus_only in [True, False]:
                    if nucleus_only:
                        gt = gt_poisson_data_tot_counts_nucl
                        other = destriped_data_tot_counts.loc[nuclear_indices]
                    else:
                        gt = gt_poisson_data_tot_counts
                        other = destriped_data_tot_counts
                    for metric in ["euclidian", "cosine"]:
                        colname = f"{metric}_distance_to_gt_take_n_counts_adjusted={take_n_counts_adjusted}_nucleus_only={nucleus_only}"
                        result_dict[colname] = distance_fun_dict_[metric](gt, other)
            return pd.Series(result_dict)

        dist_results = df_dataset_.apply(distance_to_gt, axis=1)
        df_results = pd.concat([df_results, dist_results], axis=1)
        df_results = df_results.rename(columns={"model_name": "name"})
        return df_results
