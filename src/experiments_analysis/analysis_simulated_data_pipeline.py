from pathlib import Path as P
from src.experiments_analysis.analysis import (
    summarize_poisson_sols_with_metrics_from_runs,
)
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
from src.experiments_analysis.analysis import calc_distance_destriped_data_to_gt
from pathlib import Path as P
import pandas as pd
import numpy as np


def analysis(
    output_dir,
    runs,
    dividing_by_ratio_baselines,
    not_factor_based_baseline,
    supp_baselines_dir,
    synthetic_data=True,
    fill_nans_from_original=False,
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
        fill_nans_from_original=fill_nans_from_original,
    )


def add_run_to_analysis(
    old_output_dir,
    new_output_dir,
    run_name,
    run_path,
    synthetic_data=True,
):
    """Incrementally add a single new GLUM run to an existing analysis.

    Reads existing summary DataFrames from *old_output_dir* (read-only),
    processes only the new run, and writes everything to *new_output_dir*.
    For global-structure matrices the existing .npy files are referenced
    by their original paths so nothing is copied.
    """
    import numpy as np
    from src.experiments_analysis.glum_collect_runs import collect_runs
    from src.experiments_analysis.analysis import (
        compute_sol_metrics,
        add_processed_data_paths_lightweight,
        check_nan_inf_destriped_data,
    )
    from src.experiments_analysis.analysis_utils import (
        build_df_from_sdata,
        load_gt_sol,
    )
    from src.experiments_analysis.summary_structure_preservation import (
        save_matrix_row,
        load_spatialAdata,
        difference_between_smoothed_curves,
        striping_intensity_statistics,
        striping_intensity_cyto_statistics,
    )
    from src.utilities.utils import warn_with_prefix

    old_output_dir = P(old_output_dir)
    new_output_dir = P(new_output_dir)
    new_output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Process the new run ------------------------------------------------
    df_new = collect_runs({run_name: run_path})
    df_new.reset_index("name", inplace=True)

    dataset_path = df_new["dataset_path"].iloc[0]
    cell_id_label = df_new["cell_id_label"].iloc[0]
    dataset_df = build_df_from_sdata(dataset_path, cell_id_label)
    gt_sol = load_gt_sol(P(dataset_path).parent)

    def _sol_metrics(row):
        with warn_with_prefix(f"computing sol metrics for {row['name']}: "):
            return compute_sol_metrics(row, dataset_df, gt_sol)

    records_df = df_new.apply(_sol_metrics, axis=1)
    new_poisson = pd.concat([df_new, records_df], axis=1)

    temp = pd.concat(df_new.apply(add_processed_data_paths_lightweight, axis=1).values)
    new_destriped = df_new.merge(
        right=temp, left_on="run_dir", right_index=True, how="outer"
    )
    new_destriped = check_nan_inf_destriped_data(new_destriped)

    # ---- 2. Merge with existing and save to new dir ----------------------------
    existing_poisson = pd.read_pickle(old_output_dir / "poisson_summary_df.pkl")
    existing_poisson = existing_poisson[existing_poisson["name"] != run_name]
    poisson_summary_df = pd.concat(
        [existing_poisson, new_poisson], ignore_index=True
    )
    poisson_summary_df.to_pickle(new_output_dir / "poisson_summary_df.pkl")
    poisson_summary_df.to_csv(new_output_dir / "poisson_summary_df.csv")

    existing_destriped = pd.read_pickle(old_output_dir / "destriped_summary_df.pkl")
    existing_destriped = existing_destriped[existing_destriped["name"] != run_name]
    destriped_summary_df = pd.concat(
        [existing_destriped, new_destriped], ignore_index=True
    )
    destriped_summary_df.to_pickle(new_output_dir / "destriped_summary_df.pkl")
    destriped_summary_df.to_csv(new_output_dir / "destriped_summary_df.csv")

    # ---- 3. Distance to GT (synthetic data only) ------------------------------
    if synthetic_data:
        # Load existing distances and compute only for the new run
        old_dist_dir = (
            old_output_dir / "distance_destriped_data" / "gt_destriping_method='qm'"
        )
        existing_dist_df = pd.read_pickle(old_dist_dir / "destriped_data_df.pkl")
        existing_dist_df = existing_dist_df[existing_dist_df["name"] != run_name]

        # Compute distance for just the new run + GT row
        destriped_filtered = filter_summary_df_qm(destriped_summary_df.copy())
        gt_mask = destriped_filtered["name"].str.startswith("GT_")
        new_mask = destriped_filtered["name"] == run_name
        subset = destriped_filtered.loc[gt_mask | new_mask].copy()
        new_dist_df = calc_distance_destriped_data_to_gt(
            subset,
            distance_fun_dict_=distance_fun_dict_same_index,
            gt_destriping_method="qm",
        )
        # Keep only the new run row (not the GT row)
        new_dist_df = new_dist_df[new_dist_df["name"] == run_name]

        destriped_data_df = pd.concat(
            [existing_dist_df, new_dist_df], ignore_index=True
        )
        dist_dir = (
            new_output_dir / "distance_destriped_data" / "gt_destriping_method='qm'"
        )
        dist_dir.mkdir(exist_ok=True, parents=True)
        destriped_data_df.to_pickle(dist_dir / "destriped_data_df.pkl")
        destriped_data_df.to_csv(dist_dir / "destriped_data_df.csv", index=False)

    # ---- 4. Global structure: save new matrix + re-run stats -------------------
    gs_folder = new_output_dir / "global_structure_analysis"
    gs_folder.mkdir(exist_ok=True, parents=True)
    matrices_folder = gs_folder / "destriping_matrices"
    matrices_folder.mkdir(exist_ok=True, parents=True)

    destriped_gs = filter_summary_df_qm(destriped_summary_df.copy())
    destriped_gs["fitting_method"] = destriped_gs["name"]
    destriped_gs["destriping_method"] = destriped_gs["destriping_method"].replace(
        np.nan, ""
    )
    destriped_gs["name"] = destriped_gs["name"].replace(
        {"original__dividing_by_factors": "original"}
    )

    # Save the new run's matrix
    new_idx = destriped_gs.loc[destriped_gs["fitting_method"] == run_name].index[0]
    original_data = load_spatialAdata(dataset_path)
    save_matrix_row(destriped_gs.loc[new_idx], original_data, matrices_folder)

    # Build updated matrix CSV: keep old paths, add new path
    old_gs = old_output_dir / "global_structure_analysis" / "destriping_matrices"
    existing_csv = pd.read_csv(old_gs / "df_results_path_matrices.csv")
    existing_csv = existing_csv[existing_csv["name"] != run_name]
    new_entry = pd.DataFrame(
        [
            {
                "name": run_name,
                "index_in_df_results": int(new_idx),
                "path_destriped_n_counts_matrix": str(
                    matrices_folder / f"matrix_{new_idx}.npy"
                ),
            }
        ]
    )
    updated_csv = pd.concat([existing_csv, new_entry], ignore_index=True)
    updated_csv.to_csv(matrices_folder / "df_results_path_matrices.csv", index=False)

    # Re-run stats incrementally: only compute for the new run, concat with old
    df_input = pd.read_csv(matrices_folder / "df_results_path_matrices.csv")
    df_new_row = df_input[df_input["name"] == run_name]
    old_gs_folder = old_output_dir / "global_structure_analysis"

    refs = ["original"]
    if synthetic_data:
        refs.append("GT_nbinom_sol")

    # -- difference_between_smoothed_curves: run only for the new comp key --
    gs_plots = gs_folder / "plots_global_structure"
    gs_plots.mkdir(exist_ok=True, parents=True)
    old_gs_stats = old_gs_folder / "plots_global_structure" / "statistics_global_structure.csv"
    existing_stats = pd.read_csv(old_gs_stats)
    existing_stats = existing_stats[existing_stats["comp"] != run_name]
    # Compute only for the new run as comp (needs ref matrices too)
    difference_between_smoothed_curves(df_input, [run_name], refs, gs_plots)
    new_stats = pd.read_csv(gs_plots / "statistics_global_structure.csv")
    combined_stats = pd.concat([existing_stats, new_stats], ignore_index=True)
    combined_stats.to_csv(gs_plots / "statistics_global_structure.csv", index=False)

    # -- striping_intensity_statistics: run only on new row --
    si_folder = gs_folder / "striping_intensity"
    si_folder.mkdir(exist_ok=True, parents=True)
    old_si = old_gs_folder / "striping_intensity" / "striping_intensity_statistics.csv"
    existing_si = pd.read_csv(old_si)
    existing_si = existing_si[existing_si["name"] != run_name]
    striping_intensity_statistics(df_new_row, si_folder, dataset_path, normalized=True)
    new_si = pd.read_csv(si_folder / "striping_intensity_statistics.csv")
    combined_si = pd.concat([existing_si, new_si], ignore_index=True)
    combined_si.to_csv(si_folder / "striping_intensity_statistics.csv", index=False)

    # -- striping_intensity_cyto_statistics: run only on new row --
    cyto_si_folder = gs_folder / "cyto_striping_intensity"
    cyto_si_folder.mkdir(exist_ok=True, parents=True)
    old_cyto_si = old_gs_folder / "cyto_striping_intensity" / "cyto_striping_intensity_statistics.csv"
    existing_cyto_si = pd.read_csv(old_cyto_si)
    existing_cyto_si = existing_cyto_si[existing_cyto_si["name"] != run_name]
    striping_intensity_cyto_statistics(
        df_new_row, cyto_si_folder, dataset_path, cell_id_label, normalized=True
    )
    new_cyto_si = pd.read_csv(cyto_si_folder / "cyto_striping_intensity_statistics.csv")
    combined_cyto_si = pd.concat([existing_cyto_si, new_cyto_si], ignore_index=True)
    combined_cyto_si.to_csv(cyto_si_folder / "cyto_striping_intensity_statistics.csv", index=False)
