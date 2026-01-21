from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path as P
from src.experiments_analysis.summary_structure_preservation import (
    cytoplasm_select_matrix,
    data_select_matrix,
    smoothed_lineplot_from_key,
    striping_intensity_all,
)
from src.utilities.custom_imshow import custom_imshow
from src.utilities.matplotlib_utils import pad_axes_in_points


def barplots_distance_to_gt(
    metrics_df,
    methods,
    color_dict,
    name_error_in_corrected_counts,
    name_error_in_stripe_factors,
    outpath=None,
    axes=None,
):
    df = metrics_df.copy()
    df = df[df["name"].isin(methods)]
    df["name"] = pd.Categorical(df["name"], categories=methods, ordered=True)

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 2))
    else:
        fig = axes[0].get_figure()

    sns.barplot(
        data=df,
        x="name",
        y=name_error_in_corrected_counts,
        hue="name",
        palette=color_dict,
        errorbar="sd",
        ax=axes[0],
        log_scale=True,
        log=True,
        alpha=1.0,
        errcolor="grey",
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].set_title(name_error_in_corrected_counts)
    axes[0].tick_params(axis="x", rotation=30)

    df_selection = df.loc[~df[name_error_in_stripe_factors].isna()]
    df_selection["name"] = df_selection["name"].cat.remove_unused_categories()
    print(df_selection.name.unique())
    sns.barplot(
        data=df_selection,
        x="name",
        y=name_error_in_stripe_factors,
        hue="name",
        palette=color_dict,
        errorbar="se",
        ax=axes[1],
        log_scale=True,
        log=True,
        alpha=1.0,
        errcolor="grey",
    )
    axes[1].set_title(name_error_in_stripe_factors)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")

    if not fig.get_constrained_layout():
        fig.tight_layout()
    if not (outpath is None):
        outpath = P(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath)
    return axes


def compare_destriped_data_plots(
    to_plot,
    global_structure_dir,
    region_slice,
    model_name_replacement_dict=None,
    axes=None,
    colorbar_same_scale=True,
    **imshow_kwargs,
):
    matrices_output_folder = P(global_structure_dir) / "destriping_matrices"
    df_matrices = pd.read_csv(matrices_output_folder / "df_results_path_matrices.csv")
    all_values = []

    model_name_replacement_dict_inv = {
        val: key for key, val in model_name_replacement_dict.items()
    }

    for name in to_plot:
        matrix_path = df_matrices.query(
            f"name == '{model_name_replacement_dict_inv.get(name, name)}'"
        )["path_destriped_n_counts_matrix"].item()
        matrix = np.load(matrix_path)
        sub = matrix[region_slice]
        all_values.append(sub)

    stacked = np.concatenate([m[~np.isnan(m)] for m in all_values])
    vmin_global, vmax_global = stacked.min(), stacked.max()
    if colorbar_same_scale:
        vmin = vmin_global
        vmax = vmax_global
        print("Using vmin, vmax =", vmin_global, vmax_global)
    else:
        vmin = None
        vmax = None

    if axes is None:
        fig, axes = plt.subplots(len(to_plot), 1, figsize=(8, 3 * len(to_plot)))
    for ax, name in zip(axes, to_plot):
        name_ = model_name_replacement_dict_inv.get(name, name)
        matrix_path = df_matrices.query(f"name == '{name_}'")[
            "path_destriped_n_counts_matrix"
        ].item()
        matrix = np.load(matrix_path)
        sub = matrix[region_slice]

        custom_imshow(
            sub, axis=ax, na_color="black", vmin=vmin, vmax=vmax, **imshow_kwargs
        )
        ax.set_title(name)
    return axes


def get_matrix(name, global_structure_dir, model_name_replacement_dict):
    matrices_output_folder = P(global_structure_dir) / "destriping_matrices"
    df_matrices = pd.read_csv(matrices_output_folder / "df_results_path_matrices.csv")
    model_name_replacement_dict_inv = {
        val: key for key, val in model_name_replacement_dict.items()
    }
    matrix_path = df_matrices.query(
        f"name == '{model_name_replacement_dict_inv.get(name, name)}'"
    )["path_destriped_n_counts_matrix"].item()
    matrix = np.load(matrix_path)
    return matrix


def striping_intensity_region_df(
    output_dir, region_slice, to_plot, model_name_replacement_dict
):
    output_path = P(output_dir) / "destriped_summary_df.pkl"
    destriped_summary_df = pd.read_pickle(output_path)
    path_dataset = pd.unique(destriped_summary_df["dataset_path"])[0]
    nucl_label = pd.unique(destriped_summary_df["cell_id_label"])[0]
    global_structure_analysis_folder = P(output_dir) / "global_structure_analysis"

    rows = []
    for cyto_select in [True, False]:
        if cyto_select:
            matrix_select = cytoplasm_select_matrix(path_dataset, nucl_label)[
                region_slice
            ]
        else:
            matrix_select = data_select_matrix(path_dataset)[region_slice]

        for name in to_plot:
            mat = get_matrix(
                name, global_structure_analysis_folder, model_name_replacement_dict
            )[region_slice]

            striping_intensity = striping_intensity_all(
                mat, matrix_select, normalized=True
            )["striping_intensity_tot"]

            rows.extend(
                [
                    {
                        "name": name,
                        "striping intensity": striping_intensity,
                        "cyto_select": cyto_select,
                    }
                ]
            )

    df = pd.DataFrame(rows)
    return df


def striping_intensity_quantification_region_barplot(
    output_dir,
    region,
    to_plot,
    color_dict,
    model_name_replacement_dict,
    cyto_select,
    axes=None,
):
    df = striping_intensity_region_df(
        output_dir, region, to_plot, model_name_replacement_dict
    )
    if cyto_select:
        print("cytoplasmic striping intensity")
    else:
        print("overall striping intensity")

    if axes is None:
        fig, axes = plt.subplots(figsize=(3.4, 3))
    ax = sns.barplot(
        data=df.query("cyto_select == @cyto_select"),
        x="name",
        y="striping intensity",
        hue="name",
        palette=color_dict,
        ax=axes,
    )

    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")

    return axes


def striping_intensity_quantification_region_barplot_all(
    output_dir,
    region,
    region_name,
    to_plot,
    color_dict,
    model_name_replacement_dict,
    main_publi_output_folder,
):
    for cyto_select in [True, False]:
        axes = striping_intensity_quantification_region_barplot(
            output_dir,
            region,
            to_plot,
            color_dict,
            model_name_replacement_dict,
            cyto_select,
        )

        plt.savefig(
            P(main_publi_output_folder)
            / f"{region_name}__violin_striping_quantification_{cyto_select=}.pdf"
        )
        plt.show()


def intensity_profile_in_region_violinplot(
    region,
    to_plot,
    global_structure_analysis_folder,
    color_dict,
    model_name_replacement_dict,
    axes=None,
):
    rows = []
    for name in to_plot:
        mat = get_matrix(
            name, global_structure_analysis_folder, model_name_replacement_dict
        )

        # Flatten to a 1D list of values for the violin distribution
        vals = np.asarray(mat[region]).ravel()

        # Optional: drop NaNs/Infs if they can appear
        vals = vals[np.isfinite(vals)]

        rows.extend({"name": name, "value": v} for v in vals)

    df = pd.DataFrame(rows)
    if axes is None:
        fig, axes = plt.subplots(figsize=(3.4, 3))
    ax = sns.violinplot(
        data=df,
        x="name",
        y="value",
        inner="quartile",
        cut=0,
        hue="name",
        palette=color_dict,
        ax=axes,
    )

    ax.set_xlabel("")
    ax.set_ylabel("k")
    plt.xticks(rotation=45, ha="right")

    return axes


def slice_to_rect(rs, cs, shape=None):
    if shape is None:
        r0 = rs.start
        r1 = rs.stop
        c0 = cs.start
        c1 = cs.stop
    else:
        H, W = shape[:2]

        # Convert slice -> concrete (start, stop) in array coords
        r0, r1, _ = rs.indices(H)  # handles None / negative
        c0, c1, _ = cs.indices(W)

    # Rectangle wants (x,y) = (col,row), plus width/height
    return (c0, r0, c1 - c0, r1 - r0)


def region_overview_plot(
    name,
    regions,
    colors_region,
    abbr_region,
    global_structure_dir,
    model_name_replacement_dict,
    norm,
    height_colorbar,
    imshow_kwargs=None,
    rectangle_kwargs=None,
    axes=None,
    colorbar_on=True,
):
    matrix = get_matrix(name, global_structure_dir, model_name_replacement_dict)

    shape = matrix.shape
    rectangle_kwargs = {} if (rectangle_kwargs is None) else rectangle_kwargs
    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs

    all_slice = (slice(None, None), slice(None, None))
    if axes is None:
        fig, axes = plt.subplots(1, figsize=(3.4, 4))
    else:
        fig = axes.get_figure()
    to_plot = ["original"]
    compare_destriped_data_plots(
        to_plot,
        global_structure_dir,
        all_slice,
        model_name_replacement_dict,
        axes=[axes],
        colorbar_on=False,
        aspect="equal",
        **imshow_kwargs,
    )

    axes.images[0].set_norm(norm)

    axes.axis("off")
    axes.set_title("")

    for region_name, slice_ in regions.items():
        color = colors_region[region_name]
        rectangle_kwargs_ = {
            "fill": False,
            "linewidth": 2,
            "color": color,
            "alpha": 0.8,
            **rectangle_kwargs,
        }
        x, y, w, h = slice_to_rect(slice_[0], slice_[1], shape)
        axes.add_patch(Rectangle((x, y), w, h, **rectangle_kwargs_))
        axes.text(x, y, abbr_region[region_name], va="bottom", color=color, alpha=0.8)

    plt.tight_layout()
    h_cbar = height_colorbar / fig.get_size_inches()[1]
    fig.subplots_adjust(left=0, right=1, bottom=0.1, top=0.98, hspace=0, wspace=0)

    if colorbar_on:
        cax = fig.add_axes([0.1, 0.05, 0.8, h_cbar])
        cbar = fig.colorbar(axes.get_images()[0], cax=cax, orientation="horizontal")
        cbar.set_label("counts")
    return axes


def plot_compromise_striping_intensity_global_structure_alteration(
    comparison_table,
    offset_dict=None,
    colors=None,
    annotation=True,
    markers=None,
    ax=None,
    legend_=True,
):
    if colors:
        fig_width = 5
    else:
        fig_width = 3.4
    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, 2.4))

    if colors is None:
        color = "black"
        palette = None
        hue = None
        legend = False
    else:
        hue = "model"
        color = None
        palette = colors
        legend = "auto"

    if markers is None:
        style = None
        markers = True
    else:
        style = "model"
        markers = markers
    if offset_dict is None:
        offset_dict = {}

    legend = legend and legend_

    axis = sns.scatterplot(
        data=comparison_table,
        x="striping intensity",
        y="global structure alteration",
        color=color,
        hue=hue,
        palette=palette,
        legend=legend,
        style=style,
        markers=markers,
        ax=ax,
    )
    linear_width_x = (
        comparison_table["striping intensity"]
        .loc[comparison_table["striping intensity"] > 0]
        .min()
    )
    factor_linear_width = 1
    #
    # e = np.floor(np.log10(np.abs(linear_width_x))).astype(int)
    # print(e)
    # linear_width_x = np.power(10.0, e)
    axis.set_xscale(
        "symlog", linthresh=linear_width_x * factor_linear_width, linscale=0.25
    )
    # axis.set_xscale("asinh", linear_width=linear_width_x * factor_linear_width)
    linear_width_y = (
        comparison_table["global structure alteration"]
        .loc[comparison_table["global structure alteration"] > 0]
        .min()
    )
    # e = np.floor(np.log10(np.abs(linear_width_y))).astype(int)
    # print(e)
    # linear_width_y = np.power(10.0, e)
    axis.set_yscale(
        "symlog", linthresh=linear_width_y * factor_linear_width, linscale=0.25
    )
    # axis.set_yscale("asinh", linear_width=linear_width_y * factor_linear_width)

    max_distance = comparison_table["global structure alteration"].max()
    # axis.set_ylim((-linear_width_y / 10, max_distance * 1.5))

    # min_x_decade = np.floor(np.log10(comparison_table["striping intensity"].min()))
    # min_x = 10**min_x_decade * 0.9
    # max_x = comparison_table["striping intensity"].max() * 1.5
    axis.set_xlim(
        (
            comparison_table["striping intensity"].min(),
            comparison_table["striping intensity"].max(),
        )
    )
    axis.set_ylim(
        comparison_table["global structure alteration"].min(),
        comparison_table["global structure alteration"].max(),
    )

    if annotation:
        for _, row in comparison_table.iterrows():
            axis.annotate(
                row["model"],
                xy=(
                    row["striping intensity"],
                    row["global structure alteration"],
                ),  # point
                xytext=offset_dict.get(
                    row["model"], (0, 0)
                ),  # offset: 3px right, 3px up
                textcoords="offset points",
                ha="left",
                va="bottom",  # label orientation
                fontsize=9,
            )

    if not (colors is None) and legend:
        plt.legend(loc="center left", bbox_to_anchor=(1.1, 0.5), frameon=True)

    axis.grid()
    plt.tight_layout()
    pad_axes_in_points(axis, pad_left=5, pad_right=50, pad_bottom=5, pad_top=20)
    plt.tight_layout()
    return axis


def compromise_striping_intensity_global_structure_alteration(
    cyto,
    global_dir_path,
    output_folder,
    model_name_replacement_dict,
    offset_dict=None,
    to_plot=None,
    annotation=True,
    colors=None,
    markers=None,
    ax=None,
    colors_legend=True,
):
    if cyto:
        print(f"striping intensity in cytoplasm")
    else:
        print(f"striping intensity in cytoplasm + nucleus")
    if not cyto:
        striping_intensity_table_path = (
            P(global_dir_path)
            / "striping_intensity"
            / "striping_intensity_statistics.csv"
        )

        path_table = (
            P(output_folder) / "striping_intensity-global_structure_alteration.csv"
        )

    else:
        striping_intensity_table_path = (
            P(global_dir_path)
            / "cyto_striping_intensity"
            / "cyto_striping_intensity_statistics.csv"
        )

        path_table = (
            P(output_folder) / "cyto_striping_intensity-global_structure_alteration.csv"
        )

    striping_intensity_table = pd.read_csv(striping_intensity_table_path)
    distance_to_original_smoothed_path = (
        P(global_dir_path)
        / "plots_global_structure"
        / "statistics_global_structure.csv"
    )
    distance_to_original_smoothed_table = pd.read_csv(
        distance_to_original_smoothed_path
    )

    distance_to_original_smoothed_table_sel = distance_to_original_smoothed_table.query(
        "(lane_name == 'rowscolumns') and "
        "(operation_name == 'sum') and "
        "(ref == 'original')"
    ).set_index("comp")["difference"]

    striping_intensity_sel = striping_intensity_table.set_index("name")[
        "striping_intensity_tot"
    ]

    comparison_table = pd.DataFrame.from_dict(
        {
            "striping intensity": striping_intensity_sel,
            "global structure alteration": distance_to_original_smoothed_table_sel,
        }
    ).reset_index(drop=False, names="model")

    comparison_table["model"] = comparison_table["model"].replace(
        model_name_replacement_dict
    )

    # comparison table to csv

    comparison_table.to_csv(path_table)

    # plot
    if not (to_plot is None):
        # comparison_table = comparison_table.loc[comparison_table.model.isin(to_plot)]
        comparison_table = (
            comparison_table.set_index("model").loc[to_plot].reset_index()
        )

    axis = plot_compromise_striping_intensity_global_structure_alteration(
        comparison_table,
        offset_dict=offset_dict,
        colors=colors,
        annotation=annotation,
        markers=markers,
        ax=ax,
        legend_=colors_legend,
    )

    return axis


def compromise_striping_intensity_global_structure_alteration_cyto_all(
    global_dir_path,
    output_folder,
    model_name_replacement_dict,
    offset_dict=None,
    to_plot=None,
    annotation=True,
    colors=None,
    markers=None,
    cyto_all=None,
):
    if cyto_all is None:
        cyto_all = [True, False]
    all_figs_axes = []
    for cyto in cyto_all:
        if cyto:
            print(f"striping intensity in cytoplasm")
        else:
            print(f"striping intensity in cytoplasm + nucleus")
        if not cyto:
            path_figure = (
                P(output_folder) / "striping_intensity-global_structure_alteration.pdf"
            )
        else:
            path_figure = (
                P(output_folder)
                / "cyto_striping_intensity-global_structure_alteration.pdf"
            )

        axis = compromise_striping_intensity_global_structure_alteration(
            cyto,
            global_dir_path,
            output_folder,
            model_name_replacement_dict,
            offset_dict=offset_dict,
            to_plot=to_plot,
            annotation=annotation,
            colors=colors,
            markers=markers,
        )

        # plt.savefig(path_figure, bbox_inches="tight")
        plt.savefig(path_figure)
        all_figs_axes.append(axis)

    return all_figs_axes


def barplot_global_structure_alteration(
    global_dir_path,
    output_folder=None,
    model_name_replacement_dict=None,
    to_plot=None,
    colors=None,
    ax=None,
):
    if not (to_plot is None):
        to_plot = [x for x in to_plot if x != "original"]

    distance_to_original_smoothed_path = (
        P(global_dir_path)
        / "plots_global_structure"
        / "statistics_global_structure.csv"
    )
    distance_to_original_smoothed_table = pd.read_csv(
        distance_to_original_smoothed_path
    )

    distance_to_original_smoothed_table_sel = distance_to_original_smoothed_table.query(
        "(lane_name == 'rowscolumns') and "
        "(operation_name == 'sum') and "
        "(ref == 'original')"
    )

    distance_to_original_smoothed_table_sel = (
        distance_to_original_smoothed_table_sel.rename(
            columns={"difference": "global structure alteration", "comp": "model"}
        )
    )

    distance_to_original_smoothed_table_sel[
        "model"
    ] = distance_to_original_smoothed_table_sel["model"].replace(
        model_name_replacement_dict
    )
    if not (to_plot is None):
        distance_to_original_smoothed_table_sel = (
            distance_to_original_smoothed_table_sel.query("model in @to_plot")
        )

    if colors is None:
        palette = None
        hue = None
    else:
        hue = "model"
        palette = colors

    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
    axis = sns.barplot(
        data=distance_to_original_smoothed_table_sel,
        x="model",
        y="global structure alteration",
        hue=hue,
        palette=palette,
        ax=ax,
    )
    linear_width_y = (
        distance_to_original_smoothed_table_sel["global structure alteration"]
        .loc[distance_to_original_smoothed_table_sel["global structure alteration"] > 0]
        .min()
    )
    # e = np.floor(np.log10(np.abs(linear_width_y))).astype(int)
    # print(e)
    # linear_width_y = np.power(10.0, e)
    axis.set_yscale("symlog", linthresh=linear_width_y, linscale=0.25)

    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    pad_axes_in_points(axis, pad_left=0, pad_right=0, pad_bottom=0, pad_top=20)
    if not (output_folder is None):
        path_figure = P(output_folder) / "barplot_global_structure_alteration.pdf"
        plt.savefig(path_figure)
    return axis


def global_structure_plot(
    input_folder,
    output_folder,
    color_dict=None,
    operation_name="sum",
    model_name_replacement_dict=None,
    to_plot=None,
    linestyle_dict=None,
    broken_axes_1_specs=None,
    broken_axes_2_specs=None,
    ncol_legend=3,
    axes_not_broken=None,
):
    matrices_folder = P(input_folder / "destriping_matrices")
    df = pd.read_csv(matrices_folder / "df_results_path_matrices.csv")
    if not (model_name_replacement_dict is None):
        df["name"] = df["name"].replace(model_name_replacement_dict)
    if to_plot is None:
        to_plot = df["name"].tolist()
    if linestyle_dict is None:
        linestyle_dict = {}

    k = 100

    axis_1_broken = not (broken_axes_1_specs is None)
    axis_2_broken = not (broken_axes_2_specs is None)
    no_broken_axis = (broken_axes_1_specs is None) and (broken_axes_2_specs is None)

    if no_broken_axis:
        if axes_not_broken is None:
            fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.4 * 2))
        else:
            fig = axes_not_broken[-1].get_figure()
            axes = axes_not_broken
    else:
        fig = plt.figure(figsize=(3.4, 2.4 * 2))
        sps1, sps2 = GridSpec(2, 1, figure=fig)
        if not (broken_axes_1_specs is None):
            bax_1 = brokenaxes(**broken_axes_1_specs, subplot_spec=sps1)
        else:
            bax_1 = fig.add_subplot(sps1)
        if not (broken_axes_2_specs is None):
            bax_2 = brokenaxes(**broken_axes_2_specs, subplot_spec=sps2)
        else:
            bax_2 = fig.add_subplot(sps2)
        axes = [bax_1, bax_2]

    for axis, lane_name in zip([1, 0], ["row", "column"]):
        ax = axes[axis]
        for name in to_plot:
            index = df.query(f"name == '{name}'").index
            smoothed_lineplot_from_key(
                df,
                index,
                operation_name,
                k=k,
                axis=axis,
                ax=ax,
                label=name,
                alpha=0.5,
                lw=1,
                color=color_dict[name],
                ls=linestyle_dict.get(name, "solid"),
            )
            # ax.set_title(lane_name)
            if no_broken_axis:
                ax.set_xlabel(f"{lane_name} index")
                ax.set_ylabel(f"counts")
            else:
                ax.set_xlabel(f"{lane_name} index", labelpad=20)
                ax.set_ylabel(f"counts", labelpad=40)

    if no_broken_axis:
        ref_axis = axes[-1]
    else:
        if axis_2_broken:
            ref_axis = bax_1.axs[1]
        else:
            ref_axis = axes[-1]

    handles, labels = ref_axis.get_legend_handles_labels()

    box = ref_axis.get_position()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=ncol_legend,
        frameon=True,
    )

    for legline in legend.get_lines():
        legline.set_linewidth(3)

    plt.tight_layout()

    if not (no_broken_axis):
        [x.remove() for x in bax_1.diag_handles]
        bax_1.draw_diags()
        [x.remove() for x in bax_2.diag_handles]
        bax_2.draw_diags()

        plt.tight_layout()

    plt.savefig(P(output_folder) / "global_structure_sum.pdf", bbox_inches="tight")

    return axes


def check_which_results_in_to_plot(
    to_plot, global_dir_path, model_name_replacement_dict
):
    striping_intensity_table_path = (
        P(global_dir_path) / "striping_intensity" / "striping_intensity_statistics.csv"
    )
    striping_intensity_table = pd.read_csv(striping_intensity_table_path)
    names_results = (
        striping_intensity_table["name"].replace(model_name_replacement_dict).tolist()
    )
    return [x for x in to_plot if (x in names_results)]


def global_structure_plot_all_methods(
    global_structure_analysis_folder, supp_publi_output_folder
):
    from src.experiments_analysis.plots_ismb import (
        model_name_replacement_dict,
        color_dict,
    )

    to_plot = [
        "ours",
        "ours_N.I",
        "ours_P2.I",
        "bin-level norm.",
        "original",
        "b2c",
        "b2c-sym",
        "b2c-sym-nucl",
        "b2c-sym-med",
        "b2c-sym-med-nucl",
        "MRSE",
        "test_extra_entry",
    ]

    linestyle_dict = {
        "b2c-sym-nucl": "dotted",
        "b2c-sym-med-nucl": "dotted",
        "ours_N.I": "dotted",
        "ours_P2.I": "dashed",
    }

    markers_dict = {
        "b2c-sym-nucl": "D",
        "b2c-sym-med-nucl": "D",
        "ours_N.I": "D",
        "ours_P2.I": "s",
    }
    markers_dict_n = {key: markers_dict.get(key, "o") for key in to_plot}
    markers_dict = markers_dict_n

    to_plot_ = check_which_results_in_to_plot(
        to_plot, global_structure_analysis_folder, model_name_replacement_dict
    )

    compromise_striping_intensity_global_structure_alteration_cyto_all(
        global_structure_analysis_folder,
        supp_publi_output_folder,
        model_name_replacement_dict,
        offset_dict=None,
        to_plot=to_plot_,
        annotation=False,
        colors=color_dict,
        markers=markers_dict,
    )

    axis = global_structure_plot(
        global_structure_analysis_folder,
        supp_publi_output_folder,
        color_dict=color_dict,
        model_name_replacement_dict=model_name_replacement_dict,
        linestyle_dict=linestyle_dict,
        to_plot=to_plot_,
    )

    plt.show()

    axis = barplot_global_structure_alteration(
        global_structure_analysis_folder,
        supp_publi_output_folder,
        model_name_replacement_dict,
        to_plot_,
        colors=color_dict,
    )
    plt.show()


def add_curves_to_right(axes, to_plot, df, color_dict):
    y_min = df.loc[df.name.isin(to_plot), "mean"].min()
    y_max = df.loc[df.name.isin(to_plot), "mean"].max()
    for i, name in enumerate(to_plot):
        ax_prof = axes[i, 1]

        d = df.query("name == @name").sort_values("i").copy()
        x = d["i"].to_numpy(dtype=float)
        y = d["mean"].to_numpy(dtype=float)

        # Optional smooth curve
        # kr = KernelReg(endog=y, exog=x, var_type="c", reg_type="lc", bw=[bw])
        # xg = np.unique(x)
        # yhat, _ = kr.fit(xg)
        # print(yhat.shape)
        # print(x.shape)

        ax_prof.plot(y, x, color=color_dict[name], lw=1.5)
        ax_prof.grid(False)
        ax_prof.set_xlabel("")  # optional
        ax_prof.set_ylabel("")  # keep shared y from image
        ax_prof.set_xlim(y_min, y_max)

        ax_prof.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
        )

        for sp in ["right", "bottom"]:
            ax_prof.spines[sp].set_visible(False)


def align_left_right_plots(fig, axes):
    fig.canvas.draw()
    for i, _ in enumerate(axes[:, 0]):
        ax_img = axes[i, 0]
        pos = ax_img.get_position()
        extent = ax_img.get_window_extent()
        ax_prof = axes[i, 1]
        p2 = ax_prof.get_position()
        inv = fig.transFigure.inverted()
        bbox_fig = extent.transformed(inv)
        ax_prof.set_position([p2.x0, pos.y0, p2.width, bbox_fig.height])

    fig.canvas.draw()


def get_summary_count_per_row_for_region(
    output_dir,
    region_slice,
    to_plot,
    model_name_replacement_dict,
    fun="mean",
    data_select=True,
):
    output_path = P(output_dir) / "destriped_summary_df.pkl"
    global_structure_analysis_folder = P(output_dir) / "global_structure_analysis"
    destriped_summary_df = pd.read_pickle(output_path)
    path_dataset = pd.unique(destriped_summary_df["dataset_path"])[0]
    rows = []
    for name in to_plot:
        mat = get_matrix(
            name, global_structure_analysis_folder, model_name_replacement_dict
        )

        if data_select:
            data_selector = data_select_matrix(path_dataset)[region_slice]
            kwargs = {"where": data_selector}
        else:
            kwargs = {}
        vals = getattr(mat[region_slice], fun)(axis=1, **kwargs)

        rows.extend({"name": name, fun: v, "i": i} for i, v in enumerate(vals))

    df = pd.DataFrame(rows)
    return df
