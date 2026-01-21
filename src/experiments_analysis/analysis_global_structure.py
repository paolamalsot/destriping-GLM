from src.experiments_analysis.summary_structure_preservation import difference_between_smoothed_curves, make_plots_global_structure, plots_striping_intensity_statistics, save_n_counts_matrices, striping_intensity_cyto_statistics, striping_intensity_statistics
import numpy as np
import pandas as pd


from pathlib import Path as P
color_dict_default = {
    "fit theta iter": "khaki",
    "original": "yellow",
    "out + warm_start_alpha": "red",
    "out + warm_start_alpha + custom_CV_split": "green",
    "ratio_init": "blue",
}
marker_dict_default = {
        "cyto_destripe_dividing_factors_nucl_destripe_dividing_factors": "o",
        "cyto_destripe_dividing_factors_nucl_destripe_dividing_factors_qm_tot_counts": "x",
        "destripe_dividing_factors": "o",
        "": "o",
        "destripe_dividing_factors_qm_tot_counts": "x"
}


def analysis_global_structure(
    df_results,
    output_folder,
    marker_dict=None,
    color_dict=None,
    to_plot_global_structure=None,
    refs_global_structure=None,
):
    # note that it is good to add in the name both the fitting method and the destripe method
    # to_plot_global_structure: list of df_results["names"] that we want to appear on the global structure plots
    # refs_global_structure: list of names to compare difference in global structure metrics against

    if marker_dict is None:
        marker_dict = marker_dict_default

    if color_dict is None:
        color_dict = color_dict_default

    df_results["destriping_method"] = df_results["destriping_method"].replace(np.nan, "")

    index_oi_dict = {
        row["name"]: index
        for index, row in df_results.iterrows()
    }

    matrices_output_folder = P(output_folder) / "destriping_matrices"
    matrices_output_folder.mkdir(parents=True, exist_ok=True)
    save_n_counts_matrices(df_results, index_oi_dict, matrices_output_folder)

    global_structure_output_folder = P(output_folder) / "plots_global_structure"
    global_structure_output_folder.mkdir(parents=True, exist_ok=True)
    df_input = pd.read_csv(matrices_output_folder / "df_results_path_matrices.csv")

    if to_plot_global_structure is None:
        to_plot_global_structure = df_results["name"].tolist()

    if len(to_plot_global_structure)>0:

        style_dict = {
            row["name"]: {
                "color": color_dict[row["fitting_method"]],
                "marker": marker_dict[row["destriping_method"]],
                "ms": 2,
                "markevery": 100,
                "lw": 0.5,
            }
            for index, row in df_results.iterrows()
        }

        make_plots_global_structure(
            df_input,
            to_plot_global_structure.copy(),
            global_structure_output_folder,
            style_dict=style_dict,
            color_dict_=None,
        )

    comp_keys = index_oi_dict.keys()

    if refs_global_structure is None:
        refs_global_structure = ["original"]

    difference_between_smoothed_curves(
        df_input, comp_keys, refs_global_structure, global_structure_output_folder, cosine_dist=True
    )

    striping_intensity_output_folder = P(output_folder) / "striping_intensity"
    cyto_striping_intensity_output_folder = P(output_folder) / "cyto_striping_intensity"

    striping_intensity_statistics(df_input, striping_intensity_output_folder, df_results["dataset_path"].iloc[0], normalized = True)
    df_striping_intensity = pd.read_csv(
        striping_intensity_output_folder / "striping_intensity_statistics.csv"
    )

    striping_intensity_cyto_statistics(
        df_input,
        cyto_striping_intensity_output_folder,
        df_results["dataset_path"].iloc[0],
        df_results["cell_id_label"].iloc[0],
        normalized=True
    )

    df_cyto_striping_intensity = pd.read_csv(
        cyto_striping_intensity_output_folder / "cyto_striping_intensity_statistics.csv"
    )

    plots_striping_intensity_statistics(
        df_striping_intensity, striping_intensity_output_folder
    )

    plots_striping_intensity_statistics(
        df_cyto_striping_intensity, cyto_striping_intensity_output_folder
    )