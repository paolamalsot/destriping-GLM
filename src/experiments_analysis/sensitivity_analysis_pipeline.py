"""
Analysis pipeline for segmentation-sensitivity experiments.

Thin wrapper around analysis_simulated_data_pipeline.analysis():
the YAML includes reference baselines (original, b2c, b2c-sym) alongside
the sensitivity runs so that analysis() works unchanged, including
refs_global_structure=["original"].

Typical usage (see notebooks/segmentation_analysis/sensitivity_analysis.ipynb):

    from src.experiments_analysis.sensitivity_analysis_pipeline import (
        run_sensitivity_analysis,
        build_metrics_df,
        sensitivity_barplot,
    )
    run_sensitivity_analysis(output_dir, runs, dividing_by_ratio_baselines,
                             not_factor_based_baseline)
    metrics_df = build_metrics_df(output_dir)
    sensitivity_barplot(metrics_df, output_dir, global_structure_analysis_folder)
"""

from __future__ import annotations

import json
import re
import yaml
from pathlib import Path as P
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.experiments_analysis.analysis import filter_summary_df_qm
from src.experiments_analysis.analysis_plots import (
    barplots_distance_to_gt,
    check_which_results_in_to_plot,
    compare_destriped_data_plots,
    compromise_striping_intensity_global_structure_alteration_cyto_all,
    compromise_striping_intensity_global_structure_alteration_cyto_all_with_errorbars,
    global_structure_plot,
    barplot_global_structure_alteration,
    intensity_profile_in_region_violinplot,
    striping_intensity_quantification_region_barplot,
)
from src.experiments_analysis.analysis_simulated_data_pipeline import analysis as _upstream_analysis


def make_config(baselines_path: str, sensitivity_runs_path: str) -> dict:
    """Merge a baselines YAML and a sensitivity-runs YAML into one config dict."""
    baselines = yaml.safe_load(open(baselines_path))
    sensitivity = yaml.safe_load(open(sensitivity_runs_path))
    # sensitivity keys take precedence (e.g. output_dir)
    return {**baselines, **sensitivity}


def analysis(
    output_dir,
    runs,
    dividing_by_ratio_baselines,
    not_factor_based_baseline,
    supp_baselines_dir=None,
    synthetic_data=True,
):
    _upstream_analysis(
        output_dir=output_dir,
        runs=runs,
        dividing_by_ratio_baselines=dividing_by_ratio_baselines,
        not_factor_based_baseline=not_factor_based_baseline,
        supp_baselines_dir=supp_baselines_dir,
        synthetic_data=synthetic_data,
        fill_nans_from_original=True,
    )
from src.experiments_analysis.plots_ismb import (
    color_dict as _base_color_dict,
    model_name_replacement_dict as _model_name_replacement_dict,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PERCENTAGES = [1.0, 0.95, 0.90, 0.80, 0.50, 0.25, 0.10, 0.01, 0.001]


def _pct_key(pct: float) -> str:
    """YAML runs-dict key for a percentage, e.g. 0.001 -> 'pct_0p1'."""
    val = pct * 100
    if val >= 1:
        return f"pct_{int(round(val))}"
    return f"pct_{val:.1f}".replace(".", "p")


def _pct_label(pct: float) -> str:
    """Display label, e.g. 0.001 -> 'ours (0.1%)'."""
    val = pct * 100
    if val >= 1:
        return f"ours ({int(round(val))}%)"
    return f"ours ({val:.1f}%)"


SENSITIVITY_NAME_MAP = {_pct_key(pct): _pct_label(pct) for pct in PERCENTAGES}

# All name replacements to apply when loading results
_NAME_REPLACE = {
    **SENSITIVITY_NAME_MAP,
    **_model_name_replacement_dict,
    "original__dividing_by_factors": "original",
}

NAME_ERROR_IN_CORRECTED_COUNTS = "Cosine error in $k^{corr}$"
NAME_ERROR_IN_STRIPE_FACTORS = r"Log-space L2 error in $(\mathbf{h}, \mathbf{w})$"

_METRIC_COSINE = "cosine_distance_to_gt_take_n_counts_adjusted=True_nucleus_only=True"
_METRIC_HW = "distance_to_gt_poisson_sol_hw_log_euclidian"
_RENAME_METRICS = {
    _METRIC_HW: NAME_ERROR_IN_STRIPE_FACTORS,
    _METRIC_COSINE: NAME_ERROR_IN_CORRECTED_COUNTS,
}

# Display order: sensitivity runs (100% → 0.1%) then references
METHODS_ORDER = (
    [_pct_label(pct) for pct in sorted(PERCENTAGES, reverse=True)]
    + ["original", "b2c", "b2c-sym"]
)

_SEED_MARKERS = ["o", "s", "D", "^", "v"]  # one marker per seed

# Matches sensitivity display names: "ours (80%)" (subsample) and "ours (p=50%)" (merge/split).
_SENSITIVITY_PATTERN = re.compile(r"ours \((?:p=)?(\d+(?:\.\d+)?)%\)")

# For each experiment type: True means higher p → lighter color (more modification).
# subsample is the opposite: higher p = less modification → darker.
_HIGHER_P_LIGHTER: dict[str, bool] = {
    "merge": True,
    "split": True,
    "subsample": False,
}

# Fixed display order for baseline/reference methods (appended after sensitivity runs).
_BASELINES_ORDER = [
    "bin-level norm.",
    "original",
    "b2c",
    "b2c-sym",
    "ours",
    "ours (collapse label)",
#    "b2c-sym-c_mean",
]

TO_PLOT_REMOVE = ["GT_poisson_sol", "expected_spatial_data_wo_stripes", "b2c-sym-c_mean"]

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _sensitivity_color_dict() -> dict:
    """Gradient of the 'ours' base color, darkening with lower percentage."""
    base_rgb = mcolors.to_rgb(_base_color_dict["ours"])
    alphas = np.linspace(0.35, 1.0, len(PERCENTAGES))
    return {
        _pct_label(pct): tuple(1 - a * (1 - c) for c in base_rgb)
        for pct, a in zip(sorted(PERCENTAGES, reverse=True), alphas)
    }


def _extract_trailing_number(name: str) -> float:
    """Return the last numeric value in a name string, or 0 if none found."""
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)$", name)
    return float(m.group(1)) if m else 0.0


def build_color_dict(names: list | None = None, type: str | None = None) -> dict:
    """Combined color dict: reference methods + sensitivity runs.

    Parameters
    ----------
    names:
        Full list of names to plot (e.g. the ``to_plot`` list).  Names matching
        ``ours (p=`` are extracted and colored as a gradient sorted by p value,
        light→dark or dark→light depending on ``type`` (see ``_HIGHER_P_LIGHTER``).
        Non-matching names are ignored here (they get colors from the base/sensitivity dicts).
    type:
        Experiment type — ``"merge"``, ``"split"``, or ``"subsample"``.  Controls
        gradient direction for the seed runs.  ``None`` defaults to subsample
        behaviour (higher p → darker).
    """
    d = dict(_base_color_dict)
    d.update(_sensitivity_color_dict())
    if names:
        extra_names = [n for n in names if _SENSITIVITY_PATTERN.match(n)]
        base_rgb = mcolors.to_rgb(_base_color_dict["ours"])
        higher_p_lighter = _HIGHER_P_LIGHTER.get(type, False)
        sorted_names = sorted(
            extra_names,
            key=lambda n: float(_SENSITIVITY_PATTERN.match(n).group(1)),
        )
        # alpha 0.35 = lightest tint, 1.0 = full base color (darkest)
        alphas = np.linspace(0.35, 1.0, max(len(sorted_names), 1))
        if higher_p_lighter:
            alphas = alphas[::-1]  # higher p (last after sort) gets lowest alpha
        for name, a in zip(sorted_names, alphas):
            d[name] = tuple(1 - a * (1 - c) for c in base_rgb)
    return d


def build_marker_dict(to_plot: list, seeds: list | None = None) -> dict:
    """Return a marker dict for each name in to_plot, keyed by seed index."""
    seed_str_list = [str(s) for s in (seeds or [])]
    markers_dict = {}
    for name in to_plot:
        m = re.match(r"seed_([a-zA-Z0-9_]+)__", name)
        if m and m.group(1) in seed_str_list:
            idx = seed_str_list.index(m.group(1))
            markers_dict[name] = _SEED_MARKERS[idx % len(_SEED_MARKERS)]
        else:
            markers_dict[name] = "o"
    return markers_dict


def _p_merge_key_to_label(suffix: str) -> str:
    """Convert a p_merge suffix to a display label.

    '50' → 'ours (p=50%)', '0.01' → 'ours (p=1%)'
    """
    val = float(suffix)
    pct = int(round(val if val >= 1 else val * 100))
    return f"ours (p={pct}%)"


def _display_name(raw_name: str) -> str:
    """Centralised name mapping: _NAME_REPLACE → seed-prefixed patterns → raw."""
    if raw_name in _NAME_REPLACE:
        return _NAME_REPLACE[raw_name]
    # Handle seed-prefixed names: seed_X__<inner_name>
    m = re.match(r"seed_[^_]+__(.+)$", raw_name)
    if m:
        inner = m.group(1)
        if inner in _NAME_REPLACE:
            return _NAME_REPLACE[inner]
        m2 = re.match(r"p_(?:merge|split)_(.+)$", inner)
        if m2:
            return _p_merge_key_to_label(m2.group(1))
    return raw_name


def _extract_seed(raw_name: str) -> int | None:
    """Extract seed integer from a seed-prefixed name like ``seed_42__p_split_5``."""
    m = re.match(r"seed_(\d+)__", raw_name)
    return int(m.group(1)) if m else None


def _build_seed_run_replacement(global_dir_path) -> dict:
    """Build a name-replacement dict for seed-prefixed run names."""
    csv_path = P(global_dir_path) / "striping_intensity" / "striping_intensity_statistics.csv"
    df = pd.read_csv(csv_path)
    return {name: _display_name(name) for name in df["name"].tolist() if _display_name(name) != name}


def _infer_to_plot_for_seeds(global_dir_path, seeds, model_name_replacement_dict, seed_repl: dict | None = None) -> list:
    """Return display names to plot: seed-prefixed runs for given seeds + all baselines."""
    if seed_repl is None:
        seed_repl = _build_seed_run_replacement(global_dir_path)
    augmented = {**model_name_replacement_dict, **seed_repl}
    csv_path = P(global_dir_path) / "striping_intensity" / "striping_intensity_statistics.csv"
    df = pd.read_csv(csv_path)
    names = df["name"].replace(augmented).tolist()
    seed_strs = [str(s) for s in (seeds or [])]
    result = []
    for raw_name, display_name in zip(df["name"].tolist(), names):
        m = re.match(r"seed_([a-zA-Z0-9_]+)__", raw_name)
        if m:
            if seeds is None: #all seeds by default
                result.append(display_name)
            else:
                if seed_strs and m.group(1) in seed_strs:
                    result.append(display_name)
        else:
            if not(display_name in TO_PLOT_REMOVE):
                result.append(display_name)
    # Preserve insertion order, deduplicate
    seen = set()
    return [n for n in result if not (n in seen or seen.add(n))]


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------


def run_sensitivity_analysis_multi_seed(cfg: dict, synthetic_data = False) -> None:
    """Run a single analysis over all seeds from a merged multi-seed config (simulated data).

    For each seed, prefixes run names as ``seed_{seed}__{name}``. Shared
    ``runs_baselines`` are added unprefixed. Calls ``run_sensitivity_analysis``
    once with the flattened runs dict.
    """
    runs_baselines = cfg.get("runs_baselines", {})
    flat_runs = {}

    for seed, seed_cfg in cfg["seeds"].items():
        for name, path in seed_cfg["runs"].items():
            flat_runs[f"seed_{seed}__{name}"] = path

    # Add shared run baselines (not prefixed — they're seed-independent)
    flat_runs.update(runs_baselines)

    analysis(
        output_dir=cfg["output_dir"],
        runs=flat_runs,
        dividing_by_ratio_baselines=cfg["dividing_by_ratio_baselines"],
        not_factor_based_baseline=cfg["not_factor_based_baseline"],
        supp_baselines_dir=cfg.get("supp_baselines_dir"),
        synthetic_data=synthetic_data
    )


def print_chosen_alphas(runs_path: str) -> pd.DataFrame:
    """Print chosen alpha and boundary info for all seeds/runs in a multi-seed YAML."""
    import pickle

    cfg = yaml.safe_load(open(runs_path))
    rows = []
    for seed, seed_cfg in cfg["seeds"].items():
        for run_key, run_dir in seed_cfg["runs"].items():
            try:
                glm = pickle.load(open(P(run_dir) / "glm.pkl", "rb"))
                r = glm.regressor
                full_grid = sorted(r.param_grid["alpha"])
                rows.append({
                    "run": run_key, "seed": seed, "alpha": r.alpha_,
                    "grid_min": full_grid[0], "grid_max": full_grid[-1],
                })
            except Exception:
                pass
    alpha_df = pd.DataFrame(rows)
    alpha_df["boundary_side"] = alpha_df.apply(
        lambda r: "MIN" if r["alpha"] == r["grid_min"]
        else ("MAX" if r["alpha"] == r["grid_max"] else "ok"),
        axis=1,
    )
    print("Chosen alpha:")
    print(alpha_df.pivot(index="run", columns="seed", values="alpha").to_string())
    print("\nBoundary hit wrt full CV grid (MIN / MAX / ok):")
    print(alpha_df.pivot(index="run", columns="seed", values="boundary_side").to_string())
    return alpha_df


# ---------------------------------------------------------------------------
# Metrics loading
# ---------------------------------------------------------------------------


def build_metrics_df_multi(output_dirs: list) -> pd.DataFrame:
    """Concatenate metrics from multiple subsampling seeds.

    Returns a DataFrame with one row per (method, seed), suitable for
    seaborn's errorbar='sd' to show std across subsampling seeds.
    """
    return pd.concat(
        [build_metrics_df(d) for d in output_dirs], ignore_index=True
    )


def build_metrics_df(output_dir: str) -> pd.DataFrame:
    """Load GT-distance metrics from output_dir and apply all name replacements."""
    poisson_df = pd.read_pickle(P(output_dir) / "poisson_summary_df.pkl")
    destriped_df = pd.read_pickle(
        P(output_dir)
        / "distance_destriped_data"
        / "gt_destriping_method='qm'"
        / "destriped_data_df.pkl"
    )
    destriped_df = filter_summary_df_qm(destriped_df)

    poisson_df["name"] = poisson_df["name"].map(_display_name)
    destriped_df["name"] = destriped_df["name"].map(_display_name)
    # remove the GT_poisson_sol (diff. from GT_nbinom_sol)
    to_remove = "GT_poisson_sol"
    destriped_df = destriped_df.query("name != @to_remove")

    return pd.merge(
        poisson_df[["name", _METRIC_HW]],
        destriped_df[["name", _METRIC_COSINE]],
        on="name",
    ).rename(columns=_RENAME_METRICS)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------



def sensitivity_barplot(
    metrics_df: pd.DataFrame,
    output_dir: str,
    global_structure_analysis_folder: str,
    type: str | None = None,
    seeds: list | None = None,
    to_plot: list | None = None,
) -> tuple:
    """Barplot of both GT-distance metrics for sensitivity runs + reference methods.

    Parameters
    ----------
    metrics_df:
        DataFrame returned by ``build_metrics_df`` / ``build_metrics_df_multi``.
    output_dir:
        Directory where ``sensitivity_barplot.pdf`` is written.
    global_structure_analysis_folder:
        Path to the global structure analysis folder.
    type:
        Experiment type (``"merge"``, ``"split"``, ``"subsample"``).
    seeds:
        Seeds to include.  ``None`` includes all.
    to_plot:
        Explicit list of display names to plot.  ``None`` auto-infers.
    """

    methods_order, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder, type, seeds, to_plot
    )

    fig, axes = plt.subplots(1, 2, figsize=(3.4 * 2, 3))
    output_path = P(output_dir) / "sensitivity_barplot.pdf"
    barplots_distance_to_gt(
        metrics_df,
        methods_order,
        color_dict,
        NAME_ERROR_IN_CORRECTED_COUNTS,
        NAME_ERROR_IN_STRIPE_FACTORS,
        outpath=None,
        axes=axes,
    )
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path)
    return fig, axes


def compute_tsne_log_stripe_factors(
    output_dir: str,
    seeds: list[int] | None = None,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Load log-stripe factor feature vectors and run t-SNE.

    Loads ``poisson_summary_df.pkl`` from *output_dir*.  The seed is
    extracted from each row's name (e.g. ``seed_42__p_split_5``).
    Baselines without a seed prefix are assigned seed 0.

    If *seeds* is ``None`` all seeds found in the data are included;
    otherwise only rows whose extracted seed is in *seeds* (plus
    baselines) are kept.

    Returns
    -------
    tsne_embedding : np.ndarray, shape (n_points, 2)
    labels : list[str]  — method name per point
    seeds_out : list[int] — seed per point
    """
    from sklearn.manifold import TSNE

    from src.destriping.sol import Sol
    from src.experiments_analysis.analysis_dist import log_offset_dict

    EPS = log_offset_dict["h"]
    df = pd.read_pickle(P(output_dir) / "poisson_summary_df.pkl")

    # Identify the reference (original) row by normalising names.
    ref_h_idx = ref_w_idx = None
    for _, row in df.iterrows():
        if pd.isna(row["poisson_sol_path"]):
            continue
        if _display_name(row["name"]) == "original":
            ref_sol = Sol.load(row["poisson_sol_path"])
            ref_h_idx = ref_sol.h.index
            ref_w_idx = ref_sol.w.index
            break

    records = []
    for _, row in df.iterrows():
        if pd.isna(row["poisson_sol_path"]):
            continue
        seed = _extract_seed(row["name"])
        is_baseline = seed is None
        if seed is None:
            seed = 0
        if seeds is not None and not is_baseline and seed not in seeds:
            continue
        sol = Sol.load(row["poisson_sol_path"])
        h = sol.h.reindex(ref_h_idx).fillna(1.0) if ref_h_idx is not None else sol.h
        w = sol.w.reindex(ref_w_idx).fillna(1.0) if ref_w_idx is not None else sol.w
        log_h = np.log(h.values + EPS)
        log_w = np.log(w.values + EPS)
        records.append({
            "name": row["name"],
            "seed": seed,
            "features": np.concatenate([log_h, log_w]),
        })

    X = np.stack([r["features"] for r in records])
    labels = [r["name"] for r in records]
    seeds_out = [r["seed"] for r in records]

    perplexity = min(10, len(X) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    tsne_embedding = tsne.fit_transform(X)
    print(f"t-SNE done: {X.shape[0]} points, feature dim {X.shape[1]}")
    return tsne_embedding, labels, seeds_out


def compute_log_stripe_distance_to_ours(
    output_dir: str,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Compute normalized L2 distance in log(h+ε), log(w+ε) space to "ours".

    Feature construction mirrors compute_tsne_log_stripe_factors (same EPS,
    same reindexing to the "original" reference). The reference method is
    whichever row maps to ``"ours"`` after ``_NAME_REPLACE``, generalising
    across merge/split/subsample experiments.

    The seed is extracted from each row's name (e.g. ``seed_42__...``).
    Baselines without a seed prefix are assigned seed 0.  If *seeds* is
    ``None`` all seeds are included; otherwise only matching seeds (plus
    baselines) are kept.

    Returns a DataFrame with columns ``["name", "seed", "distance"]`` using
    display names (via ``_NAME_REPLACE``).  The caller / plot function handles
    aggregation (mean) and error bars (std).
    """
    from src.destriping.sol import Sol
    from src.experiments_analysis.analysis_dist import log_offset_dict

    EPS = log_offset_dict["h"]
    df = pd.read_pickle(P(output_dir) / "poisson_summary_df.pkl")

    ref_h_idx = ref_w_idx = None
    for _, row in df.iterrows():
        if pd.isna(row["poisson_sol_path"]):
            continue
        if _display_name(row["name"]) == "original":
            ref_sol = Sol.load(row["poisson_sol_path"])
            ref_h_idx = ref_sol.h.index
            ref_w_idx = ref_sol.w.index
            break

    records = []
    for _, row in df.iterrows():
        if pd.isna(row["poisson_sol_path"]):
            continue
        seed = _extract_seed(row["name"])
        is_baseline = seed is None
        if seed is None:
            seed = 0
        if seeds is not None and not is_baseline and seed not in seeds:
            continue
        sol = Sol.load(row["poisson_sol_path"])
        h = sol.h.reindex(ref_h_idx).fillna(1.0) if ref_h_idx is not None else sol.h
        w = sol.w.reindex(ref_w_idx).fillna(1.0) if ref_w_idx is not None else sol.w
        records.append({
            "name": _display_name(row["name"]),
            "seed": seed,
            "features": np.concatenate([np.log(h.values + EPS), np.log(w.values + EPS)]),
        })

    try:
        ref_vec = next(r["features"] for r in records if r["name"] == "ours")
    except StopIteration:
        raise ValueError('"ours" not found in any of the provided output_dirs')

    dim = len(ref_vec)
    rows = []
    for r in records:
        dist = 0.0 if r["name"] == "ours" else np.linalg.norm(r["features"] - ref_vec) / np.sqrt(dim)
        rows.append({"name": r["name"], "seed": r["seed"], "distance": dist})

    df = pd.DataFrame(rows)
    return df[df["name"] != "ours"].reset_index(drop=True)


def plot_log_stripe_distance_to_ours(
    distance_df: pd.DataFrame,
    global_structure_analysis_folder: str,
    type: str | None = None,
    seeds: list | None = None,
    to_plot: list | None = None,
) -> plt.Figure.axes:
    """Barplot of normalized L2 distance in log-space to "ours".

    Parameters
    ----------
    distance_df:
        Per-seed DataFrame with columns ``["name", "seed", "distance"]``
        returned by ``compute_log_stripe_distance_to_ours``.
    global_structure_analysis_folder:
        Path to the global structure analysis folder.
    type:
        Experiment type (``"merge"``, ``"split"``, ``"subsample"``).
        Controls ordering and color gradient direction.
    seeds:
        Seeds to include.  ``None`` includes all.
    to_plot:
        Explicit list of display names to plot.  ``None`` auto-infers.
    """
    agg = distance_df.groupby("name", as_index=False)["distance"].agg(["mean", "std"])
    agg.columns = ["name", "mean", "std"]
    agg["std"] = agg["std"].fillna(0.0)

    ordered_names, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder, type, seeds, to_plot
    )
    agg = agg.set_index("name").loc[[n for n in ordered_names if n in agg["name"].values]].reset_index()
    colors = [color_dict.get(n, "gray") for n in agg["name"]]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(range(len(agg)), agg["mean"], yerr=agg["std"], color=colors, capsize=3)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(agg["name"], rotation=45, ha="right")
    ax.set_ylabel(r"")
    ax.set_title(r"Log-space L2 distance to ours in $(\mathbf{h}, \mathbf{w})$")
    ax.set_yscale("log")
    plt.tight_layout()
    return ax


def plot_tsne_log_stripe_factors(
    tsne_embedding: np.ndarray,
    labels: list[str],
    seeds: list[int],
    extra_names: list[str] | None = None,
) -> plt.Figure:
    """Scatter plot of a 2-D t-SNE embedding of log-stripe factors.

    Points are colored by method name (using build_color_dict) and shaped by
    seed. Returns the figure.
    """
    color_dict = build_color_dict(extra_names)
    seed_vals = sorted(set(seeds))
    markers = ["o", "s", "^", "D", "v"]
    seed_marker = {s: markers[i % len(markers)] for i, s in enumerate(seed_vals)}

    fig, ax = plt.subplots(figsize=(8, 6))
    seen_names: dict = {}
    seen_seeds: dict = {}
    for i, (x, y) in enumerate(tsne_embedding):
        name = labels[i]
        seed = seeds[i]
        color = color_dict.get(name, "gray")
        marker = seed_marker[seed]
        ax.scatter(x, y, color=color, marker=marker, s=80, zorder=3)
        ax.text(x, y, name, fontsize=7, ha="left", va="bottom")
        if name not in seen_names:
            seen_names[name] = ax.scatter([], [], color=color, label=name, s=60)
        if seed not in seen_seeds:
            seen_seeds[seed] = ax.scatter(
                [], [], color="k", marker=marker, label=f"seed {seed}", s=60
            )

    ax.legend(
        handles=list(seen_names.values()) + list(seen_seeds.values()),
        fontsize=7,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    ax.set_title(
        r"t-SNE of $\log(\mathbf{h}+\varepsilon)$ and $\log(\mathbf{w}+\varepsilon)$"
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    return fig


def order(to_plot: list, type: str | None = None, baselines_first: bool = False) -> list:
    """Order to_plot: ours, then sensitivity runs (less → more modified), then other baselines.

    Sensitivity runs are names matching ``ours (p=`` — sorted by their p value
    from least modified to most modified (direction depends on ``type``).
    Baselines follow in ``_BASELINES_ORDER`` order.

    If *baselines_first* is ``True`` the order is reversed: baselines first,
    then ours, then sensitivity runs.
    """
    seed_names = [n for n in to_plot if _SENSITIVITY_PATTERN.match(n)]
    baseline_names = [n for n in to_plot if (not _SENSITIVITY_PATTERN.match(n)) and (n != "ours")]
    reverse_sort = not _HIGHER_P_LIGHTER.get(type, False)  # subsample: high p first
    seed_names = sorted(
        seed_names,
        key=lambda n: float(_SENSITIVITY_PATTERN.match(n).group(1)),
        reverse=reverse_sort,
    )
    baseline_names = sorted(
        baseline_names,
        key=lambda n: _BASELINES_ORDER.index(n) if n in _BASELINES_ORDER else len(_BASELINES_ORDER),
    )

    merge_only_baselines = ["b2c-sym-c_mean", "ours (collapse label)"]
    if type != "merge":
        baseline_names = [e for e in baseline_names if not(e in merge_only_baselines)]

    ours_list = ["ours" ] if ("ours" in to_plot) else []

    if baselines_first:
        return baseline_names + ours_list + seed_names
    return ours_list + seed_names + baseline_names


def global_structure_plot_all_methods(
    global_structure_analysis_folder,
    output_dir,
    type: str | None = None,
    seeds: list | None = None,
    to_plot: list | None = None,
):
    to_plot, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder, type, seeds, to_plot
    )

    markers_dict = build_marker_dict(to_plot, seeds)

    compromise_striping_intensity_global_structure_alteration_cyto_all(
        global_structure_analysis_folder,
        output_dir,
        aug_dict,
        offset_dict=None,
        to_plot=to_plot,
        annotation=False,
        colors=color_dict,
        markers=markers_dict,
    )

    plt.show()

    axis = global_structure_plot(
        global_structure_analysis_folder,
        output_dir,
        color_dict=color_dict,
        model_name_replacement_dict=aug_dict,
        linestyle_dict={},
        to_plot=to_plot,
    )

    plt.show()

    axis = barplot_global_structure_alteration(
        global_structure_analysis_folder,
        output_dir,
        aug_dict,
        to_plot,
        colors=color_dict,
    )
    plt.show()

def get_figure_params(global_structure_analysis_folder, type, seeds, to_plot, baselines_first: bool = False):
    from src.experiments_analysis.plots_ismb import (
        model_name_replacement_dict,
    )

    seed_repl = _build_seed_run_replacement(global_structure_analysis_folder)
    aug_dict = {**model_name_replacement_dict, **seed_repl}

    if to_plot is None:
        to_plot = _infer_to_plot_for_seeds(
            global_structure_analysis_folder,
            seeds,
            model_name_replacement_dict,
            seed_repl=seed_repl,
        )

    to_plot = order(to_plot, type, baselines_first=baselines_first)

    color_dict = build_color_dict(to_plot, type=type)

    to_plot = check_which_results_in_to_plot(
        to_plot, global_structure_analysis_folder, aug_dict
    )

    return to_plot, aug_dict, color_dict


def _load_striping_intensity_plot_df(
    global_structure_analysis_folder,
    model_name_replacement_dict,
    to_plot=None,
    cyto: bool = False,
) -> pd.DataFrame:
    """Load striping intensity data formatted for bar plotting."""
    if cyto:
        csv_path = (
            P(global_structure_analysis_folder)
            / "cyto_striping_intensity"
            / "cyto_striping_intensity_statistics.csv"
        )
    else:
        csv_path = (
            P(global_structure_analysis_folder)
            / "striping_intensity"
            / "striping_intensity_statistics.csv"
        )

    df = pd.read_csv(csv_path).rename(
        columns={"name": "model", "striping_intensity_tot": "striping intensity"}
    )
    df["model"] = df["model"].replace(model_name_replacement_dict)

    if to_plot is not None:
        df = df[df["model"].isin(to_plot)].copy()
        if not df.empty:
            df["model"] = pd.Categorical(df["model"], categories=to_plot, ordered=True)
            df = df.sort_values("model")

    return df


def barplot_global_structure_alteration_(global_structure_analysis_folder,
    output_dir,
    type: str | None = None,
    seeds: list | None = None,
    to_plot: list | None = None,
    ax = None
  ):
    
    to_plot, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder, type, seeds, to_plot
    )

    axis = barplot_global_structure_alteration(
        global_structure_analysis_folder,
        output_dir,
        aug_dict,
        to_plot,
        colors=color_dict,
        ax=ax
    )
    plt.tight_layout()
    return axis


def barplot_striping_intensity(
    global_structure_analysis_folder,
    output_dir,
    type: str | None = None,
    seeds: list | None = None,
    to_plot: list | None = None,
    ax=None,
    cyto: bool = False,
):
    """Barplot of striping intensity for sensitivity runs and baselines."""
    to_plot, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder, type, seeds, to_plot
    )
    df = _load_striping_intensity_plot_df(
        global_structure_analysis_folder,
        aug_dict,
        to_plot=to_plot,
        cyto=cyto,
    )
    if df.empty:
        raise ValueError("No striping-intensity results available for the requested plot.")
    #plot_order = [name for name in to_plot if name != "original"] if to_plot is not None else None

    hue = "model" if color_dict is not None else None
    palette = color_dict if color_dict is not None else None

    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 2.8))
    
    axis = sns.barplot(
        data=df,
        x="model",
        y="striping intensity",
        hue=hue,
        palette=palette,
        order=to_plot,
        hue_order=to_plot,
        ax=ax,
    )

    positive = df["striping intensity"].loc[df["striping intensity"] > 0]
    if not positive.empty:
        linear_width_y = positive.min()
        axis.set_yscale("symlog", linthresh=linear_width_y, linscale=0.25)

    axis.set_xlabel("")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
    axis.set_ylabel("Striping Int.")
    plt.tight_layout()

    if output_dir is not None:
        filename = (
            "barplot_cyto_striping_intensity.pdf"
            if cyto
            else "barplot_striping_intensity.pdf"
        )
        plt.savefig(P(output_dir) / filename)

    return axis


def compromise_plot_with_errorbars(
    global_structure_analysis_folder,
    output_dir,
    type: str | None = None,
    seeds: list | None = None,
    to_plot: list | None = None,
    alpha: float = 0.8,
    cyto_all: list | None = None,
    ax = None
):
    """Compromise scatter plot with mean±std error bars across seeds.

    Same setup as ``global_structure_plot_all_methods`` but calls the
    errorbars variant and does NOT need markers.
    """
    to_plot, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder, type, seeds, to_plot
    )

    compromise_striping_intensity_global_structure_alteration_cyto_all_with_errorbars(
        global_structure_analysis_folder,
        output_dir,
        aug_dict,
        offset_dict=None,
        to_plot=to_plot,
        annotation=False,
        colors=color_dict,
        alpha=alpha,
        cyto_all = cyto_all,
        ax = ax
    )

    return ax


# ---------------------------------------------------------------------------
# Cell-size inputs loader
# ---------------------------------------------------------------------------


def _get_merge_subdirs(run_dir: P) -> dict:
    """Return {p_merge_float: merge_subdir_path} for a merging run directory."""
    result = {}
    for d in run_dir.glob("merge_p*"):
        meta = json.loads((d / "metadata.json").read_text())
        result[float(meta["p_merge"])] = d
    return result


def load_cell_size_inputs(cfg: dict) -> tuple:
    """Assemble inputs for plot_mean_cell_size from a merge sensitivity config.

    Returns
    -------
    labels_by_p : dict[float, np.ndarray]
        {0.0: original labels, p1: merged labels, ...}
    coords : pd.DataFrame
        DataFrame with columns 'array_row' and 'array_col'.
    stats_df : pd.DataFrame
        DataFrame with columns 'p' and 'pct_merged'.
    """
    from src.spatialAdata.loading import load_spatialAdata

    subdirs = {}
    merging_run_dir = None

    for run_dir in cfg["runs"].values():
        hydra_config_dir = next(P(run_dir).rglob(".hydra"))
        hydra_cfg = yaml.safe_load((hydra_config_dir / "config.yaml").read_text())

        if "merging_run_dir" not in hydra_cfg:
            continue  # skip p_merge_0 (baseline, no merging)
        merging_run_dir = P(hydra_cfg["merging_run_dir"])
        p_merge_float = float(hydra_cfg["p_merge"])
        subdirs[p_merge_float] = _get_merge_subdirs(merging_run_dir)[p_merge_float]

    if merging_run_dir is None:
        raise ValueError("No merge run found in cfg (all entries lack merging_run_dir).")

    merging_cfg = yaml.safe_load((merging_run_dir / ".hydra/config.yaml").read_text())
    path_data = merging_cfg["dataset"]["path_data"]
    cell_id_label = merging_cfg["dataset"]["cell_id_label"]
    sdata = load_spatialAdata(path_data)
    sdata.add_array_coords_to_obs()
    coords = sdata.obs[["array_row", "array_col"]].copy()

    labels_by_p = {0.0: sdata.obs[cell_id_label].values}
    for p, merge_dir in sorted(subdirs.items()):
        labels = pd.read_parquet(merge_dir / "labels.parquet").loc[:, cell_id_label]
        labels_by_p[p] = labels.values

    rows = []
    for p, merge_dir in subdirs.items():
        meta = json.loads((merge_dir / "metadata.json").read_text())
        pct_merged = (meta["n_nuclei_before"] - meta["n_nuclei_after"]) / meta["n_nuclei_before"] * 100
        rows.append({"p": p, "pct_merged": pct_merged})
    stats_df = pd.DataFrame(rows)

    return labels_by_p, coords, stats_df

##### FINAL PLOTS


def arrange_legend_2_columns(handles, labels):
    """Reorder handles/labels for a 2-column legend: 'ours' variants left, others right."""
    left_col = lambda l: (_SENSITIVITY_PATTERN.match(l)) or ("ours" == l)
    ours = [(h, l) for h, l in zip(handles, labels) if left_col(l)]
    others = [(h, l) for h, l in zip(handles, labels) if not left_col(l)]
    n_rows = max(len(ours), len(others))
    ours += [(plt.Line2D([], [], alpha=0), "")] * (n_rows - len(ours))
    others += [(plt.Line2D([], [], alpha=0), "")] * (n_rows - len(others))
    ordered = [*ours, *others]
    ordered_handles, ordered_labels = zip(*ordered)
    return ordered_handles, ordered_labels


def panel_global_structure_alteration_striping_intensity(
    global_structure_analysis_folder, experiment_type, cyto_all = True):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(5 / 0.8, 2.4 * 3))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 0.3, 1])
    ax_compromise = fig.add_subplot(gs[0, :])
    ax_legend = fig.add_subplot(gs[1, :])
    ax_bar_left = fig.add_subplot(gs[2, 0])
    ax_bar_right = fig.add_subplot(gs[2, 1])

    compromise_plot_with_errorbars(
        global_structure_analysis_folder,
        None,
        type=experiment_type,
        seeds=None,
        cyto_all=[cyto_all],
        ax=ax_compromise,
    )
    # Move legend below compromise plot in 2 columns
    handles, labels = ax_compromise.get_legend_handles_labels()
    ax_compromise.get_legend().remove()
    ordered_handles, ordered_labels = arrange_legend_2_columns(handles, labels)
    ax_legend.legend(
        ordered_handles, ordered_labels, loc="center", frameon=True, ncol=2
    )
    ax_legend.axis("off")

    barplot_global_structure_alteration_(
        global_structure_analysis_folder,
        type=experiment_type,
        output_dir=None,
        ax=ax_bar_left,
    )
    barplot_striping_intensity(
        global_structure_analysis_folder,
        None,
        ax=ax_bar_right,
        type=experiment_type,
        cyto=cyto_all,
    )
    plt.tight_layout()
    return fig


def real_data_plots(output_dir, figures_output_dir, experiment_type):

    global_structure_analysis_folder = P(output_dir) / "global_structure_analysis"
    axes = panel_global_structure_alteration_striping_intensity(
        global_structure_analysis_folder, experiment_type
    )

    plt.tight_layout()
    plt.savefig(
        figures_output_dir / "global_structure_alteration_striping_intensity_panel.pdf"
    )
    plt.show()

    dist_df = compute_log_stripe_distance_to_ours(
        output_dir=output_dir,
    )

    ax = plot_log_stripe_distance_to_ours(dist_df, global_structure_analysis_folder, type=experiment_type, seeds=None)
    plt.savefig(figures_output_dir / "log_space_distance_to_ours.pdf")
    plt.show()


_DEFAULT_BASELINES_KEEP = ["original", "b2c", "b2c-sym-c_mean", "ours (collapse label)"]


def _get_panel_params(output_dir, experiment_type, seed, baselines_keep=None):
    """Shared setup for panel plotting functions.

    Returns ``(to_plot, aug_dict, color_dict, global_structure_analysis_folder)``
    with baselines filtered to *baselines_keep*.
    """
    if baselines_keep is None:
        baselines_keep = _DEFAULT_BASELINES_KEEP

    global_structure_analysis_folder = P(output_dir) / "global_structure_analysis"

    to_plot, aug_dict, color_dict = get_figure_params(
        global_structure_analysis_folder,
        type=experiment_type,
        seeds=[seed],
        to_plot=None,
        baselines_first=True,
    )

    to_plot = [
        n for n in to_plot
        if _SENSITIVITY_PATTERN.match(n) or n == "ours" or n in baselines_keep
    ]

    return to_plot, aug_dict, color_dict, global_structure_analysis_folder


def destriped_data_panel(
    output_dir,
    experiment_type: str,
    region_slice,
    seed: int = 42,
    cyto_select: bool = False,
    n_cols: int = 3,
    figsize_per_cell: tuple[float, float] = (3.4, 3.4),
    baselines_keep: list | None = None,
):
    """3-column panel of destriped data images with striping intensity barplot.

    All methods are laid out in reading order (row-major) across *n_cols*
    columns, with baselines first (via ``baselines_first=True``).
    A striping-intensity barplot spans the bottom row.
    """
    from math import ceil
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import PowerNorm

    to_plot, aug_dict, color_dict, global_structure_analysis_folder = _get_panel_params(
        output_dir, experiment_type, seed, baselines_keep
    )

    n_image_rows = ceil(len(to_plot) / n_cols)

    #w = figsize_per_cell[0] * n_cols
    w = 7.009
    h = figsize_per_cell[1] * (n_image_rows + 0.8) / (figsize_per_cell[0] * n_cols) * w
    #h = figsize_per_cell[1] * (n_image_rows + 0.8)

    fig = plt.figure(figsize=(w, h))
    gs = gridspec.GridSpec(
        n_image_rows + 1,
        n_cols,
        figure=fig,
        height_ratios=[1] * n_image_rows + [0.8],
    )

    # Image axes in reading order
    image_axes = []
    for idx in range(len(to_plot)):
        row, col = divmod(idx, n_cols)
        image_axes.append(fig.add_subplot(gs[row, col]))

    # Hide unused trailing cells
    for idx in range(len(to_plot), n_image_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    compare_destriped_data_plots(
        to_plot,
        global_structure_analysis_folder,
        region_slice,
        aug_dict,
        axes=image_axes,
        colorbar_same_scale=False,
    )

    # Post-process image axes
    for ax in image_axes:
        ax.images[0].set_norm(PowerNorm(gamma=0.5))
        ax.axis("off")

    # Barplot spanning all columns
    barplot_ax = fig.add_subplot(gs[n_image_rows, :])
    striping_intensity_quantification_region_barplot(
        output_dir,
        region_slice,
        to_plot,
        color_dict,
        aug_dict,
        cyto_select,
        axes=barplot_ax,
    )

    fig.tight_layout()
    return fig


def intensity_violinplot(
    output_dir,
    experiment_type: str,
    region_slice,
    seed: int = 42,
    baselines_keep: list | None = None,
    ax=None,
):
    """Violin plot of intensity profiles for sensitivity runs and baselines."""
    to_plot, aug_dict, color_dict, global_structure_analysis_folder = _get_panel_params(
        output_dir, experiment_type, seed, baselines_keep
    )

    ax = intensity_profile_in_region_violinplot(
        region_slice,
        to_plot,
        global_structure_analysis_folder,
        color_dict,
        aug_dict,
        axes=ax,
    )
    ax.set_yscale("symlog", linthresh=1, linscale=0.1)
    ax.set_ylim(0, 500)
    return ax


def convergence_barplot(
    output_dir,
    experiment_type: str,
    seed: int | None = None,
    baselines_keep: list | None = None,
    ax=None,
):
    """Barplot of convergence percentage across seeds for ours + sensitivity runs.

    An instance is "converged" when both ``converged`` and
    ``theta_iter_converged_`` are ``True`` in the poisson summary DataFrame.
    """
    to_plot, aug_dict, color_dict, _ = _get_panel_params(
        output_dir, experiment_type, seed=seed or 42, baselines_keep=baselines_keep
    )

    # Keep only "ours" and sensitivity runs
    to_plot = [n for n in to_plot if _SENSITIVITY_PATTERN.match(n) or n == "ours"]

    df = pd.read_csv(P(output_dir) / "poisson_summary_df.csv")
    df["display_name"] = df["name"].replace(aug_dict)
    df = df[df["display_name"].isin(to_plot)].copy()
    df["is_converged"] = df["converged"].astype(bool) & df["theta_iter_converged_"].astype(bool)

    pct = df.groupby("display_name")["is_converged"].mean() * 100
    pct = pct.reindex(to_plot)

    if ax is None:
        _, ax = plt.subplots(figsize=(3.4, 2.8))

    sns.barplot(
        x=pct.index,
        y=pct.values,
        hue=pct.index,
        palette=color_dict,
        order=to_plot,
        hue_order=to_plot,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Converged (%)")
    ax.set_ylim(0, 105)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    return ax


def simulation_data_plots(output_dir, figures_output_dir, experiment_type):

    global_structure_analysis_folder = P(output_dir) / "global_structure_analysis"
    axes = panel_global_structure_alteration_striping_intensity(
        global_structure_analysis_folder, experiment_type, cyto_all=False
    )

    plt.tight_layout()
    plt.savefig(
        figures_output_dir / "global_structure_alteration_striping_intensity_panel.pdf"
    )
    plt.show()

    metrics_df = build_metrics_df(output_dir)
    print(metrics_df)

    fig, axes = sensitivity_barplot(metrics_df, figures_output_dir, global_structure_analysis_folder, type=experiment_type)
    plt.show()

def qualitative_plots_human_lymph_node(output_dir, figures_output_dir, experiment_type):

    # destriping panel

    region_slice = (slice(3100, 3250), slice(2675, 2825))
    fig = destriped_data_panel(
        output_dir,
        experiment_type=experiment_type,
        region_slice=region_slice,
        seed=42,
        cyto_select=False,
    )
    output_path = figures_output_dir / "destriping_region_B.pdf"
    plt.savefig(output_path)
    plt.show()

    # counts distribution

    bottom_right_region = (slice(2200, 3000), slice(3000, None))
    ax = intensity_violinplot(
        output_dir,
        experiment_type=experiment_type,
        region_slice=bottom_right_region,
        seed=42,
    )
    output_path = figures_output_dir / "counts_distribution_region_A.pdf"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
