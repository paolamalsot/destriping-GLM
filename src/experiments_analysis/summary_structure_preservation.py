import numpy as np
from pathlib import Path as P
from src.experiments_analysis.analysis_utils import (
    get_destriped_tot_counts,
    get_index_gt_destriped,
)
from src.spatialAdata.loading import load_spatialAdata, img_2D_from_vals
import numpy as np
import pandas as pd
import warnings
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.destriping.sol import Sol
from scipy.spatial.distance import cosine
from pathlib import Path as P
from src.utilities.utils import warn_with_prefix


def dataset_folder_from_folder_result(folder_results):
    # could be made more robust my using the dataset name from one df...
    folder_results_in = [
        folder for folder in folder_results.iterdir() if folder.is_dir()
    ][0]
    folder_results_in = [
        folder
        for folder in folder_results_in.iterdir()
        if (
            folder.is_dir() and ("df_results.csv" in [f.name for f in folder.iterdir()])
        )
    ][0]
    return folder_results_in


operation_dict = {
    "sum": lambda x, axis: np.nansum(x, axis=axis),
}


def load_stripe_factor(row, factor_name):
    poisson_sol_path = row["poisson_sol_path"]
    if factor_name == "h":
        return Sol.load_h(poisson_sol_path)
    elif factor_name == "w":
        return Sol.load_w(poisson_sol_path)
    raise ValueError(f"Unknown factor name: {factor_name}")


def make_smoothed_line(vals, k):
    """
    Create a smoothed line by convolving the input values with a uniform kernel of size k.
    """
    if len(vals) < k:
        raise ValueError("Length of vals must be greater than or equal to k.")
    smoothed_line = np.convolve(vals, np.ones(k) / k, mode="valid")
    return smoothed_line


def smoothed_lineplot(quant_per_x, k, ax, **kwargs):
    smoothed_line = make_smoothed_line(quant_per_x, k)
    ax.plot(np.arange(len(smoothed_line)), smoothed_line, **kwargs)
    return ax


def smoothed_lineplot_from_key(df, index, operation_name, k, axis, ax, **kwargs):
    destriped_matrix_path = df.loc[index, "path_destriped_n_counts_matrix"].item()
    matrix = np.load(destriped_matrix_path)
    vals = operation_dict[operation_name](matrix, axis=axis)
    smoothed_lineplot(vals, k, ax, **kwargs)


def smoothed_lineplot_stripe_factor_from_key(df, index, factor_name, k, ax, **kwargs):
    stripe_factor = load_stripe_factor(df.loc[index], factor_name=factor_name)
    return smoothed_lineplot(stripe_factor.values, k=k, ax=ax, **kwargs)


def smoothed_line_from_key(df, index, operation_name, k, axis):
    destriped_matrix_path = df.loc[index, "path_destriped_n_counts_matrix"].item()
    matrix = np.load(destriped_matrix_path)
    vals = operation_dict[operation_name](matrix, axis=axis)
    return make_smoothed_line(vals, k)


def base_get_index_oi_dict(df, distance_colname=None):
    df_results_with_args = pd.concat(
        [df, df.apply(lambda row: pd.Series(row["fitting_args"]), axis=1)],
        axis=1,
    )

    index_original = df_results_with_args.query(
        "(model_name == 'init') and (factors == 'ones')"
    ).index[0]

    index_init_quantiles = df_results_with_args.query(
        "(model_name == 'init') and (factors == 'quantiles')"
    ).index[0]
    index_init_quantiles_nucl = df_results_with_args.query(
        "(model_name == 'init') and (factors == 'quantiles_nucl')"
    ).index[0]
    try:
        index_init_median_ratio = df_results_with_args.query(
            "(model_name == 'init') and (factors == 'median_ratio')"
        ).index[0]
    except IndexError:
        print("No init_median_ratio model found in the results. Skipping it.")
        index_init_median_ratio = None

    try:
        index_b2c = df_results_with_args.query("(model_name == 'b2c')").index[0]
    except IndexError:
        print("No b2c model found in the results. Skipping it.")
        index_b2c = None

    # Create base dictionary with consistent ordering
    base_dict = [
        ("original", index_original),
        ("b2c", index_b2c),
        ("init_quantiles", index_init_quantiles),
        ("init_quantiles_nucl", index_init_quantiles_nucl),
        ("init_median_ratio", index_init_median_ratio),
    ]

    index_dict = dict(base_dict)

    return index_dict


def get_index_oi_dict(df, distance_colname=None):
    base_dict = base_get_index_oi_dict(df, distance_colname)
    ## add the ground truth
    gt_index = get_index_gt_destriped(df["dataset_path"].unique()[0], df)
    base_dict["GT"] = gt_index
    return base_dict


def save_matrix_row(row, original_data, save_dir):
    i_row = row.name
    destriped_data_counts = get_destriped_tot_counts(row, take_n_counts_adjusted=True)
    intersect_indices = original_data.index.intersection(destriped_data_counts.index)
    array_coords_ = original_data[intersect_indices].get_unscaled_coordinates("array")
    destriped_data_counts_ = destriped_data_counts.loc[intersect_indices]
    matrix = img_2D_from_vals(array_coords_, destriped_data_counts_)
    matrix_path = save_dir / f"matrix_{i_row}.npy"
    np.save(matrix_path, matrix)
    return row


def cytoplasm_select_matrix(dataset_path, nucl_label):
    data = load_spatialAdata(dataset_path)
    cytoplasm_select = np.logical_not(data.nucl_mask(nucl_label))
    data.obs["is_cyto"] = cytoplasm_select
    matrix = data.matrix_from_label("is_cyto")
    matrix[pd.isna(matrix)] = False
    return matrix.astype(bool)


color_dict = {
    "original": "yellow",
    "GT": "blue",
    "b2c": "red",
    "neighbour_loss_no_reg_within_nucl": "green",
    "neighbour_loss_best": "darkgreen",
    "neighbour_loss_no_reg_all": "chartreuse",
    "neighbour_loss_worst": "mediumspringgreen",
    "init_quantiles_nucl": "brown",
    "init_quantiles": "orange",
    "init_median_ratio": "khaki",
}
to_plot_simulation = [
    "original",
    "GT",
    "b2c",
    "neighbour_loss_no_reg_within_nucl",
    "neighbour_loss_best",
    "neighbour_loss_worst",
    "init_quantiles_nucl",
    "init_median_ratio",
]
to_plot_real_data = to_plot_simulation.copy()
to_plot_real_data.remove("GT")
to_plot_real_data.append(
    "neighbour_loss_no_reg_all"
)  # GT is not available for real data


def make_plots_global_structure(
    df, to_plot, save_dir: P, k=100, color_dict_=None, style_dict=None
):
    save_dir.mkdir(parents=True, exist_ok=True)
    if (color_dict_ is None) and (style_dict is None):
        color_dict_ = color_dict  # old behaviour
    else:
        assert (color_dict_ is None) ^ (style_dict is None)

    for axis, lane_name in zip([1, 0], ["rows", "columns"]):
        for operation_name, operation in operation_dict.items():
            fig, ax = plt.subplots()

            for name in to_plot:
                index = df.query(f"name == '{name}'").index
                if color_dict_ is not None:
                    kwargs = {"color": color_dict_[name]}
                else:
                    kwargs = style_dict[name]

                smoothed_lineplot_from_key(
                    df,
                    index,
                    operation_name,
                    k=k,
                    axis=axis,
                    ax=ax,
                    label=name,
                    alpha=0.5,
                    **kwargs,
                )
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            plt.title(
                f"Smoothed lineplot of {operation_name} of destriped matrix {lane_name} (k={k})"
            )
            plt.tight_layout()
            plt.savefig(
                save_dir / f"smoothed_lineplot_{operation_name}_{lane_name}_k{k}.pdf",
                bbox_inches="tight",
            )
            plt.close()


def rotate_labels(g):
    for ax in g.axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


def difference_between_smoothed_curves(
    df, comp_keys, reference_keys, save_dir, k=100, cosine_dist=False
):
    # calculates the L2 difference between the comp_keys and the references_keys
    # the curve can be for example curve of 99th quantile or sum etc...

    results = []

    for axis, lane_name in zip([1, 0], ["rows", "columns"]):
        for operation_name, operation in operation_dict.items():
            for ref in reference_keys:
                index = df.query(f"name == '{ref}'").index
                with warn_with_prefix(
                    f"Calculating smoothed line for {ref=}, {operation_name} and {axis=}: "
                ):
                    smoothed_line_ref = smoothed_line_from_key(
                        df, index, operation_name, k, axis
                    )
                nan_in_ref = np.isnan(smoothed_line_ref)
                smoothed_line_ref = smoothed_line_ref[~nan_in_ref]
                for comp in comp_keys:
                    index_comp = df.query(f"name == '{comp}'").index
                    with warn_with_prefix(
                        f"Calculating smoothed line for {comp=}, {operation_name} and {axis=}: "
                    ):
                        smoothed_line_comp = smoothed_line_from_key(
                            df, index_comp, operation_name, k, axis
                        )
                    smoothed_line_comp = smoothed_line_comp[~nan_in_ref]
                    # smoothed_line_comp[np.isnan(smoothed_line_comp)] = smoothed_line_original[np.isnan(smoothed_line_comp)]
                    if cosine_dist:
                        difference = cosine(smoothed_line_ref, smoothed_line_comp)
                    else:
                        difference = np.linalg.norm(
                            smoothed_line_ref - smoothed_line_comp
                        )
                    result_dict = {
                        "axis": axis,
                        "lane_name": lane_name,
                        "operation_name": operation_name,
                        "ref": ref,
                        "comp": comp,
                        "difference": difference,
                    }
                    results.append(result_dict)
    results_df = pd.DataFrame(results)

    # calculate the sum over the axes, and set for a new axis column named "global"
    new = results_df.groupby(["operation_name", "ref", "comp"]).sum().reset_index()
    new["axis"] = "global"
    results_df = pd.concat([results_df, new], ignore_index=True)
    results_df.to_csv(save_dir / "statistics_global_structure.csv", index=False)

    for operation_name, df in results_df.groupby("operation_name"):
        g = sns.catplot(
            kind="strip",
            x="comp",
            y="difference",
            row="ref",
            col="axis",
            hue="comp",
            data=df,
        )
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(90)
        plt.tight_layout()
        plt.savefig(
            save_dir / f"statistics_global_structure_{operation_name}.pdf",
            bbox_inches="tight",
        )
        plt.close()


def striping_intensity(matrix, axis, normalized=False):
    """
    Calculate the striping intensity of a matrix.
    Striping intensity along rows is defined as:
     $$\sqrt{\sum_r^(R-1){(s_r - s_{r+1})^2}}$$ sqrt of the squared difference between the total intensity of adjacent lanes.
    axis =0 calculate the difference between adjacent rows
    axis =1 calculate the difference between adjacent columns
    """
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1.")
    if axis == 0:
        other_axis = 1
    else:
        other_axis = 0

    I = (
        np.diff(np.nansum(matrix, other_axis, keepdims=True), axis=axis) ** 2
    ).sum() ** 0.5
    if normalized:
        N = np.sqrt(matrix.shape[axis] - 1) * np.nanmean(matrix)
    else:
        N = np.sqrt(matrix.shape[axis] - 1)
    return I / N


def striping_intensity_cyto(matrix, cyto_select, axis, normalized=False):
    """Calculate the striping intensity, but taking into account the cytoplasm only."
    The differences are taken into account only on connection that link 2 cytoplasmic bins"
    In mathematical terms, this gives:
    $$\sqrt{\sum_r^R(\sum_b^{B_r}(k_{rb} - k_{(r+1),b}))^2}}$$
    axis is the axis along which we calculate the differences. If axis = 1, we calc. differences between columns.
    If axis = 0, we calculate differences between rows.
    """
    if axis == 1:
        cyto_connection = cyto_select[:, 1:] & cyto_select[:, :-1]
        other_axis = 0
    elif axis == 0:
        cyto_connection = cyto_select[1:, :] & cyto_select[:-1, :]
        other_axis = 1
    else:
        raise ValueError("Axis must be 0 or 1.")

    diffs_cyto_only = np.ma.array(
        data=np.diff(matrix, axis=axis), mask=~cyto_connection
    )
    sums = np.nansum(diffs_cyto_only, other_axis, keepdims=True)
    I = (sums**2).sum() ** 0.5
    if normalized:
        N = np.sqrt(matrix.shape[axis] - 1) * np.nanmean(matrix)
    else:
        N = np.sqrt(matrix.shape[axis] - 1)
    return I / N


def striping_intensity_all(matrix, matrix_data_select, normalized):
    striping_intensity_row = striping_intensity_cyto(
        matrix, matrix_data_select, axis=0, normalized=normalized
    )
    striping_intensity_column = striping_intensity_cyto(
        matrix, matrix_data_select, axis=1, normalized=normalized
    )
    striping_intensity_tot = np.sqrt(
        striping_intensity_row**2 + striping_intensity_column**2
    )
    return pd.Series(
        {
            "striping_intensity_row": striping_intensity_row,
            "striping_intensity_column": striping_intensity_column,
            "striping_intensity_tot": striping_intensity_tot,
        }
    )


def data_select_matrix(dataset_path):
    data = load_spatialAdata(dataset_path)
    matrix = img_2D_from_vals(data.array_coords, np.ones(data.shape[0], dtype=bool))
    matrix[pd.isna(matrix)] = False
    return matrix.astype(bool)


def striping_intensity_statistics(df, output_folder, path_dataset, normalized=False):
    matrix_data_select = data_select_matrix(path_dataset)

    # for every row of the df, calculates the striping intensity
    def striping_intensity_per_row(row):
        destriped_matrix_path = row["path_destriped_n_counts_matrix"]
        matrix = np.load(destriped_matrix_path)
        return striping_intensity_all(matrix, matrix_data_select, normalized)

    striping_intensity_df = df.apply(striping_intensity_per_row, axis=1)
    df = pd.concat([df, striping_intensity_df], axis=1)
    outpath = output_folder / "striping_intensity_statistics.csv"
    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)


def striping_intensity_cyto_statistics(
    df, output_folder, path_dataset, nucl_label, normalized=False
):
    matrix_cyto_select = cytoplasm_select_matrix(path_dataset, nucl_label)

    # for every row of the df, calculates the striping intensity
    def striping_intensity_per_row(row):
        destriped_matrix_path = row["path_destriped_n_counts_matrix"]
        matrix = np.load(destriped_matrix_path)
        return striping_intensity_all(matrix, matrix_cyto_select, normalized)

    striping_intensity_df = df.apply(striping_intensity_per_row, axis=1)
    df = pd.concat([df, striping_intensity_df], axis=1)
    outpath = output_folder / "cyto_striping_intensity_statistics.csv"
    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(outpath, index=False)


def plots_striping_intensity_statistics(df_striping_intensity, output_folder):
    # do a catplot with every col a striping intensity column, compare the different methods (names in the "name" column)
    output_folder.mkdir(parents=True, exist_ok=True)
    df = pd.wide_to_long(
        df_striping_intensity,
        stubnames=["striping_intensity"],
        i="name",
        j="lane",
        sep="_",
        suffix="\\w+",
    ).reset_index()
    g = sns.catplot(
        col="lane",
        x="name",
        y="striping_intensity",
        kind="strip",
        data=df,
        log_scale=True,
    )
    g.set_xticklabels(rotation=90)
    plt.tight_layout()
    plt.savefig(output_folder / "plot_striping_intensity.pdf", bbox_inches="tight")
    plt.close()


def save_n_counts_matrices(df_results, index_oi_dict, output_folder):
    index_oi_series = pd.Series(index_oi_dict)
    index_oi_series.to_csv(output_folder / "index_oi.csv")
    # download original data
    dataset_path = df_results["dataset_path"].unique()[0]
    if len(df_results["dataset_path"].unique()) > 1:
        raise ValueError("Multiple dataset paths found in results, expected one.")

    original_data = load_spatialAdata(dataset_path)
    new_df_list = []

    for name, index in index_oi_series.items():
        print(name, index)
        save_matrix_row(df_results.loc[index], original_data, output_folder)
        new_df_list.append(
            {
                "name": name,
                "index_in_df_results": index,
                "path_destriped_n_counts_matrix": str(
                    output_folder / f"matrix_{index}.npy"
                ),
            }
        )

    new_df_list = pd.DataFrame(new_df_list)
    new_df_list.to_csv(output_folder / "df_results_path_matrices.csv", index=False)
