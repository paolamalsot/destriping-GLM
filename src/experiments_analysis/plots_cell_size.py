import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


def mean_cell_size_map(labels, coords, bin_size=1000):
    """For each bin with a valid label, compute its cell size, then return the mean cell size per bin."""
    label_series = pd.Series(labels, index=coords.index)
    mask = label_series.notna()
    label_valid = label_series[mask]
    coords_valid = coords[mask]

    cell_sizes = label_valid.map(label_valid.value_counts())

    row_bins = np.arange(coords["array_row"].min(), coords["array_row"].max() + bin_size, bin_size)
    col_bins = np.arange(coords["array_col"].min(), coords["array_col"].max() + bin_size, bin_size)

    H_sum, _, _ = np.histogram2d(
        coords_valid["array_row"].values,
        coords_valid["array_col"].values,
        bins=[row_bins, col_bins],
        weights=cell_sizes.values.astype(float),
    )
    H_count, _, _ = np.histogram2d(
        coords_valid["array_row"].values,
        coords_valid["array_col"].values,
        bins=[row_bins, col_bins],
    )
    with np.errstate(invalid="ignore"):
        return np.where(H_count > 0, H_sum / H_count, np.nan)


def plot_mean_cell_size(labels_by_p, coords, stats_df, bin_size=20, n_cols=3):
    """
    Plot mean cell size maps for a dict of p -> labels.

    Parameters
    ----------
    labels_by_p : dict[float, array-like]
        Mapping from p value to label array (include 0.0 for original).
    coords : pd.DataFrame
        DataFrame with columns 'array_row' and 'array_col'.
    stats_df : pd.DataFrame
        DataFrame with columns 'p' and 'pct_merged'.
    bin_size : int
        Spatial bin size in array coordinates.
    n_cols : int
        Number of subplot columns.
    """
    all_ps = list(labels_by_p.keys())
    n_rows = math.ceil(len(all_ps) / n_cols)
    w = 7.009
    h = 4 * n_rows / (5 * n_cols) * w
    # h = figsize_per_cell[1] * (n_image_rows + 0.8)

    fig = plt.figure(figsize=(w, h))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(w, h),
        constrained_layout=True,
    )

    all_means = [mean_cell_size_map(labels_by_p[p], coords, bin_size) for p in all_ps]

    for ax, p, mean_map in zip(axes.flatten(), all_ps, all_means):
        vmax = np.nanmax(mean_map)
        norm = LogNorm(vmin=1, vmax=vmax) if vmax > 100 else None
        im = ax.imshow(
            mean_map,
            origin="upper",
            aspect="equal",
            cmap="viridis",
            norm=norm,
            vmin=None if norm else 1,
        )
        fig.colorbar(im, ax=ax, label="", fraction=0.03, pad=0.02)
        ax.set_title(f"p = {p*100:.0f}%")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.flatten()[len(all_ps):]:
        ax.set_visible(False)

    fig.suptitle("Mean cell size [bins]", fontsize=14)
    return fig
