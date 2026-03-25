from __future__ import annotations

import logging
import os
import yaml
from pathlib import Path

import celltypist
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
from scipy.stats import rankdata, spearmanr

from src.experiments_analysis.plots_ismb import model_name_replacement_dict
from src.spatialAdata.loading import load_spatialAdata
from src.spatialAdata.spatialAdata import spatialAdata

_METHOD_RENAMES = {
    **model_name_replacement_dict,
    "ground_truth": "GT",
    "expected_spatial_data_wo_stripes": "GT (expectation)",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _resolve_run_dir(path: str | Path) -> Path:
    """Resolve a run path that may be a Hydra sweep root to the actual run dir."""
    p = Path(path)
    if (p / "multirun.yaml").exists():
        subdirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(subdirs) == 1, f"Expected 1 run subdir in sweep {p}, found {len(subdirs)}"
        return subdirs[0]
    return p


def _find_qm_destriped_df(run_dir: Path) -> str:
    """Find the qm (not sqm) destriped df.parquet inside a GLM run's destriped_data/."""
    destriped_root = run_dir / "destriped_data"
    for subdir in sorted(destriped_root.iterdir()):
        if subdir.name == ".DS_Store":
            continue
        if "qm" in subdir.name and "sqm" not in subdir.name:
            return str(subdir / "df.parquet")
    raise FileNotFoundError(f"No qm destriped data found in {destriped_root}")


def _get_hydra_dataset_info(run_dir: Path) -> tuple[str, str]:
    """Extract dataset path_data and cell_id_label from .hydra/config.yaml."""
    config_path = run_dir / ".hydra" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg["dataset"]["path_data"], cfg["dataset"]["cell_id_label"]


def _load_supp_baseline_df_path(
    supp_baselines_dir: str,
    model_name: str,
    destriping_method: str | None = None,
) -> str | None:
    """Load a destriped_data_path from supp_baselines df_baselines.csv."""
    csv_path = Path(supp_baselines_dir) / "results" / "df_baselines.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    mask = df["model_name"] == model_name
    if destriping_method is not None:
        mask &= df["destriping_method"] == destriping_method
    if not mask.any():
        logging.warning("%s not found in %s", model_name, csv_path)
        return None
    return df.loc[mask, "destriped_data_path"].iloc[0]


def load_benchmark_config(yaml_path: str | Path) -> dict:
    """
    Load a benchmark YAML config and resolve all paths.

    Returns
    -------
    dict with keys:
        ``"sdata_path"`` – bin-level spatialAdata path
        ``"cell_id_label"`` – label column for cell segmentation
        ``"methods"`` – ``{name: path_to_destriped_df_parquet}``
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    methods: dict[str, str] = {}

    # ours (GLM run — may be sweep root, has qm subdirectory)
    ours_run_dir = _resolve_run_dir(cfg["runs"]["ours"])
    methods["ours"] = _find_qm_destriped_df(ours_run_dir)
    sdata_path, cell_id_label = _get_hydra_dataset_info(ours_run_dir)

    # original (dividing_by_factors with ones)
    orig_path = Path(cfg["dividing_by_ratio_baselines"]["original"])
    methods["original"] = str(orig_path / "destriped_data" / "df.parquet")

    # 2SN
    b2c_path = _resolve_run_dir(cfg["not_factor_based_baseline"]["2SN"])
    methods["2SN"] = str(b2c_path / "destriped_data" / "df.parquet")

    # b2c-sym (dividing_99_quantile)
    ratio_baselines = cfg.get("dividing_by_ratio_baselines", {})
    if "dividing_99_quantile" in ratio_baselines:
        b2csym_path = Path(ratio_baselines["dividing_99_quantile"])
        methods["dividing_99_quantile"] = str(b2csym_path / "destriped_data" / "df.parquet")

    # ground truth references (if supp_baselines_dir is available)
    supp_dir = cfg.get("supp_baselines_dir")
    if supp_dir is not None:
        gt_df_path = _load_supp_baseline_df_path(
            supp_dir, "GT_nbinom_sol", "destripe_dividing_factors_qm_tot_counts"
        )
        if gt_df_path is not None:
            methods["ground_truth"] = gt_df_path

        gt_exp_path = _load_supp_baseline_df_path(
            supp_dir, "expected_spatial_data_wo_stripes"
        )
        if gt_exp_path is not None:
            methods["expected_spatial_data_wo_stripes"] = gt_exp_path

    # rename method keys to publication names
    methods = {_METHOD_RENAMES.get(k, k): v for k, v in methods.items()}

    return {
        "sdata_path": sdata_path,
        "cell_id_label": cell_id_label,
        "methods": methods,
    }


def rescale_sdata_expression(
    sdata: spatialAdata,
    destriped_df: pd.DataFrame,
    count_col: str = "n_counts",
) -> spatialAdata:
    """
    Rescale gene expression so each bin's total matches the destriped counts.

    Parameters
    ----------
    sdata : spatialAdata
        Original bin-level spatial data (will be copied, not mutated).
    destriped_df : DataFrame
        Indexed by bin ID, with a ``count_col`` column holding target total counts.
    count_col : str
        Column in *destriped_df* to use as target totals.

    Returns
    -------
    spatialAdata with rescaled X.
    """
    rescaled = sdata.copy()

    target_counts = destriped_df[count_col].reindex(rescaled.adata.obs.index)
    raw_counts = np.asarray(rescaled.adata.X.sum(axis=1)).flatten()

    scale_factor = np.where(raw_counts == 0, 0.0, target_counts.values / raw_counts)
    scale_factor = np.nan_to_num(scale_factor, nan=0.0)

    diag = scipy.sparse.diags(scale_factor)
    rescaled.adata.X = diag.dot(rescaled.adata.X)

    return rescaled


def aggregate_bins_to_cells(
    sdata: spatialAdata,
    labels_key: str = "labels_he",
    max_bin_distance: int = 0,
) -> spatialAdata:
    """
    Aggregate bins into cells via bin2cell (optionally expanding labels first).

    With ``max_bin_distance=0`` the expansion step is skipped and bins are
    grouped directly by ``labels_key``.
    """
    if max_bin_distance > 0:
        from bin2cell.bin2cell import expand_labels

        expanded_key = f"{labels_key}_expanded"
        expand_labels(
            sdata.adata,
            labels_key=labels_key,
            expanded_labels_key=expanded_key,
            max_bin_distance=max_bin_distance,
        )
        b2c_key = expanded_key
    else:
        b2c_key = labels_key

    cell_sdata = sdata.bin2cell(labels_key=b2c_key, labels_source_key=None)
    cell_sdata.n_counts  # populate obs["n_counts"]
    return cell_sdata


def run_cell_typing(
    sdata: spatialAdata,
    model: str = "Mouse_Whole_Brain.pkl",
) -> spatialAdata:
    """
    Run celltypist on cell-level sdata (round → normalise → log1p → annotate).

    Transfers ``predicted_labels`` and ``conf_score`` back to *sdata* (in place)
    and returns it.
    """
    tmp = sdata.copy()
    tmp.round_gene_expression()
    tmp.normalize_total(target_sum=1e4)
    tmp.log1p()

    predictions = celltypist.annotate(tmp.adata, model=model, majority_voting=False)
    pred_adata = predictions.to_adata()

    sdata.adata.obs["predicted_labels"] = pred_adata.obs["predicted_labels"]
    sdata.adata.obs["conf_score"] = pred_adata.obs["conf_score"]
    return sdata


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _shared_cell_index(sdatas: dict[str, spatialAdata]) -> pd.Index:
    """Return the intersection of cell indices across all methods."""
    indices = [s.adata.obs.index for s in sdatas.values()]
    shared = indices[0]
    for idx in indices[1:]:
        shared = shared.intersection(idx)
    return shared


def _row_spearman(X, Y):
    """Vectorised per-row Spearman correlation between two matrices.

    Both *X* and *Y* must be dense arrays of shape ``(n_cells, n_genes)``.
    Returns a 1-D array of length ``n_cells``.
    """
    rX = np.apply_along_axis(rankdata, 1, X)
    rY = np.apply_along_axis(rankdata, 1, Y)
    rX -= rX.mean(axis=1, keepdims=True)
    rY -= rY.mean(axis=1, keepdims=True)
    num = (rX * rY).sum(axis=1)
    den = np.sqrt((rX ** 2).sum(axis=1) * (rY ** 2).sum(axis=1))
    return np.where(den == 0, 0.0, num / den)


def per_cell_spearman_with_gt(
    sdatas: dict[str, spatialAdata],
    gt_key: str = "ground_truth",
) -> pd.DataFrame:
    """
    Per-cell Spearman correlation of gene expression vs ground truth.

    Returns DataFrame with columns: method, cell_id, spearman_rho, bin_count.
    """
    shared = _shared_cell_index(sdatas)
    gt_X = np.asarray(sdatas[gt_key].adata[shared].X.todense())
    bin_count = sdatas[gt_key].adata.obs.loc[shared, "bin_count"].values

    records = []
    for name, sdata in sdatas.items():
        if name == gt_key:
            continue
        method_X = np.asarray(sdata.adata[shared].X.todense())
        rho = _row_spearman(method_X, gt_X)
        df = pd.DataFrame({
            "method": name,
            "cell_id": shared,
            "spearman_rho": rho,
            "bin_count": bin_count,
        })
        records.append(df)
    return pd.concat(records, ignore_index=True)


def per_cell_agreement_with_gt(
    sdatas: dict[str, spatialAdata],
    gt_key: str = "ground_truth",
) -> pd.DataFrame:
    """
    Per-cell binary agreement of predicted_labels vs ground truth.

    Returns DataFrame with columns: method, cell_id, agreement, bin_count.
    """
    shared = _shared_cell_index(sdatas)
    gt_labels = sdatas[gt_key].adata.obs.loc[shared, "predicted_labels"].astype(str)
    bin_count = sdatas[gt_key].adata.obs.loc[shared, "bin_count"]

    records = []
    for name, sdata in sdatas.items():
        if name == gt_key:
            continue
        method_labels = sdata.adata.obs.loc[shared, "predicted_labels"].astype(str)
        df = pd.DataFrame({
            "method": name,
            "cell_id": shared,
            "agreement": (method_labels.values == gt_labels.values),
            "bin_count": bin_count.values,
        })
        records.append(df)
    return pd.concat(records, ignore_index=True)


def compare_binned_data(
    sdatas: dict[str, spatialAdata],
    small_cell_max_bins: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Compare cell-level expression across methods.

    Returns
    -------
    dict with keys:
        ``"total_counts"`` – per-cell n_counts for each method
        ``"rank_correlation"`` – pairwise Spearman rank-correlation of per-gene totals
        ``"small_cell_counts"`` – same as total_counts but only cells with <= small_cell_max_bins bins
    """
    shared = _shared_cell_index(sdatas)
    names = list(sdatas.keys())

    # total counts comparison
    tc = pd.DataFrame(
        {name: sdatas[name].adata.obs.loc[shared, "n_counts"] for name in names},
        index=shared,
    )
    tc["bin_count"] = sdatas[names[0]].adata.obs.loc[shared, "bin_count"]

    # small-cell subset
    small_mask = tc["bin_count"] <= small_cell_max_bins
    tc_small = tc.loc[small_mask].copy()

    # per-gene total expression rank correlation (across all cells)
    gene_totals = {}
    for name in names:
        X = sdatas[name].adata[shared].X
        gene_totals[name] = np.asarray(X.sum(axis=0)).flatten()

    corr_records = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            rho, pval = spearmanr(gene_totals[n1], gene_totals[n2])
            corr_records.append({"method_1": n1, "method_2": n2, "spearman_rho": rho, "p_value": pval})
    rank_corr = pd.DataFrame.from_records(corr_records)

    return {
        "total_counts": tc,
        "rank_correlation": rank_corr,
        "small_cell_counts": tc_small,
    }


def compare_cell_types(
    sdatas: dict[str, spatialAdata],
) -> dict[str, pd.DataFrame]:
    """
    Compare cell-type assignments and confidence scores across methods.

    Returns
    -------
    dict with keys:
        ``"agreement"`` – pairwise fraction of cells with identical predicted_labels
        ``"type_counts"`` – number of cells per type per method
        ``"conf_scores"`` – per-type median conf_score per method
    """
    shared = _shared_cell_index(sdatas)
    names = list(sdatas.keys())

    # pairwise agreement
    labels = {name: sdatas[name].adata.obs.loc[shared, "predicted_labels"].astype(str) for name in names}
    agreement_records = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1 :]:
            frac = (labels[n1] == labels[n2]).mean()
            agreement_records.append({"method_1": n1, "method_2": n2, "agreement_fraction": frac})
    agreement = pd.DataFrame.from_records(agreement_records)

    # type counts
    type_counts = pd.DataFrame(
        {name: sdatas[name].adata.obs.loc[shared, "predicted_labels"].value_counts() for name in names}
    ).fillna(0).astype(int)
    type_counts.index.name = "predicted_labels"

    # median confidence per type
    conf_parts = []
    for name in names:
        obs = sdatas[name].adata.obs.loc[shared]
        med = obs.groupby("predicted_labels")["conf_score"].median().rename(name)
        conf_parts.append(med)
    conf_scores = pd.concat(conf_parts, axis=1)
    conf_scores.index.name = "predicted_labels"

    return {
        "agreement": agreement,
        "type_counts": type_counts,
        "conf_scores": conf_scores,
    }


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    sdata_path: str | Path,
    methods: dict[str, str | Path],
    output_dir: str | Path = "results/my_notebooks/cell_typing_mouse_brain",
    labels_key: str = "labels_he",
    max_bin_distance: int = 0,
    cell_typing_model: str = "Mouse_Whole_Brain.pkl",
    count_col: str = "n_counts",
) -> dict[str, spatialAdata]:
    """
    Full pipeline: rescale → bin2cell → cell-type for each method.

    Parameters
    ----------
    sdata_path : path
        Path to the original bin-level spatialAdata.
    methods : dict
        ``{method_name: path_to_destriped_df_parquet}``.
    output_dir : path
        Root directory for saved outputs.
    labels_key, max_bin_distance, cell_typing_model, count_col :
        Forwarded to the individual pipeline steps.

    Returns
    -------
    dict mapping method name → cell-level spatialAdata (with cell types).
    """
    output_dir = Path(output_dir)
    logging.info("Loading original sdata from %s", sdata_path)
    sdata = load_spatialAdata(str(sdata_path))

    cell_sdatas: dict[str, spatialAdata] = {}
    paths_records = []

    for name, df_path in methods.items():
        logging.info("Processing method: %s", name)

        destriped_df = pd.read_parquet(df_path)

        # rescale
        rescaled = rescale_sdata_expression(sdata, destriped_df, count_col=count_col)

        # bin → cell
        cell_sdata = aggregate_bins_to_cells(
            rescaled, labels_key=labels_key, max_bin_distance=max_bin_distance
        )

        # save binned (pre cell-typing)
        save_dir = output_dir / "binned_sdata" / name
        save_dir.mkdir(parents=True, exist_ok=True)
        cell_sdata.save(str(save_dir))
        logging.info("  Saved binned sdata to %s", save_dir)

        # cell typing
        cell_sdata = run_cell_typing(cell_sdata, model=cell_typing_model)

        # save with cell types
        save_dir_ct = output_dir / "cell_typed_sdata" / name
        save_dir_ct.mkdir(parents=True, exist_ok=True)
        cell_sdata.save(str(save_dir_ct))
        logging.info("  Saved cell-typed sdata to %s", save_dir_ct)

        cell_sdatas[name] = cell_sdata
        paths_records.append(
            {"method": name, "binned_sdata_path": str(save_dir), "cell_typed_sdata_path": str(save_dir_ct)}
        )

    paths_df = pd.DataFrame.from_records(paths_records)
    paths_df.to_csv(output_dir / "paths.csv", index=False)
    logging.info("Saved paths to %s", output_dir / "paths.csv")

    return cell_sdatas


def run_unsupervised_clustering(
    sdata: spatialAdata,
    n_top_genes: int = 3000,
    n_comps: int = 30,
    n_neighbors: int = 30,
    leiden_resolution: float = 0.5,
    filter_percentile: float = 50.0,
) -> spatialAdata:
    """
    Unsupervised clustering via scanpy: percentile filter → normalize → HVG → PCA → neighbors → Leiden → UMAP.

    Parameters
    ----------
    filter_percentile : float
        Remove the bottom *filter_percentile* % of cells by total counts.
        Set to 0 to skip filtering.
    leiden_resolution : float
        Resolution for Leiden clustering (higher = more clusters).
    """
    adata = sdata.adata

    # percentile-based cell filtering
    n_before = adata.n_obs
    if filter_percentile > 0:
        total_counts = np.asarray(adata.X.sum(axis=1)).flatten()
        threshold = np.percentile(total_counts, filter_percentile)
        keep = total_counts >= threshold
        adata._inplace_subset_obs(keep)
        logging.info(
            "  Percentile filter (>= %.0f%%): kept %d / %d cells (threshold=%.1f counts)",
            filter_percentile, adata.n_obs, n_before, threshold,
        )
    sc.pp.filter_genes(adata, min_cells=3)

    # keep raw counts for later marker-gene inspection
    adata.raw = adata.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top_genes)
    adata_hvg = adata[:, adata.var.highly_variable].copy()

    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=n_comps)
    sc.pp.neighbors(adata_hvg, n_neighbors=n_neighbors, n_pcs=n_comps)
    sc.tl.leiden(adata_hvg, resolution=leiden_resolution)
    sc.tl.umap(adata_hvg)

    # transfer results back to the full adata
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    adata.obsp["connectivities"] = adata_hvg.obsp["connectivities"]
    adata.obsp["distances"] = adata_hvg.obsp["distances"]
    adata.uns["neighbors"] = adata_hvg.uns["neighbors"]
    adata.uns["leiden"] = adata_hvg.uns.get("leiden", {})
    adata.obs["leiden"] = adata_hvg.obs["leiden"]

    return sdata


def _cluster_one_method(
    binned_sdata_path: str,
    clustered_sdata_path: str,
    n_top_genes: int,
    n_neighbors: int,
    leiden_resolution: float,
    filter_percentile: float,
) -> spatialAdata:
    """Load a binned sdata, run clustering, save, and return it."""
    cell_sdata = load_spatialAdata(binned_sdata_path)
    cell_sdata = run_unsupervised_clustering(
        cell_sdata,
        n_top_genes=n_top_genes,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        filter_percentile=filter_percentile,
    )
    Path(clustered_sdata_path).mkdir(parents=True, exist_ok=True)
    cell_sdata.save(clustered_sdata_path)
    return cell_sdata


def run_pipeline_unsupervised(
    sdata_path: str | Path,
    methods: dict[str, str | Path],
    output_dir: str | Path,
    labels_key: str = "cell_id",
    max_bin_distance: int = 0,
    count_col: str = "n_counts",
    n_top_genes: int = 3000,
    n_neighbors: int = 30,
    leiden_resolution: float = 0.5,
    filter_percentile: float = 50.0,
    parallel: bool = False,
) -> dict[str, spatialAdata]:
    """
    Pipeline: rescale → bin2cell → percentile filter → unsupervised clustering for each method.

    Same structure as :func:`run_pipeline` but replaces celltypist with
    scanpy Leiden clustering + UMAP.

    Parameters
    ----------
    parallel : bool
        If True, run the clustering step for all methods in parallel
        (rescale + bin2cell are still sequential).
    """
    output_dir = Path(output_dir)
    logging.info("Loading original sdata from %s", sdata_path)
    sdata = load_spatialAdata(str(sdata_path))

    # --- Phase 1: rescale + bin2cell sequentially, save binned sdatas ---
    binned_paths: dict[str, str] = {}
    clustered_paths: dict[str, str] = {}
    for name, df_path in methods.items():
        logging.info("Processing method (bin2cell): %s", name)

        destriped_df = pd.read_parquet(df_path)
        rescaled = rescale_sdata_expression(sdata, destriped_df, count_col=count_col)
        cell_sdata = aggregate_bins_to_cells(
            rescaled, labels_key=labels_key, max_bin_distance=max_bin_distance
        )

        save_dir = str(output_dir / "binned_sdata" / name)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        cell_sdata.save(save_dir)
        logging.info("  Saved binned sdata to %s", save_dir)

        binned_paths[name] = save_dir
        clustered_paths[name] = str(output_dir / "clustered_sdata" / name)

    # free the large bin-level sdata before clustering
    del sdata

    # --- Phase 2: clustering (optionally parallel) ---
    cluster_kwargs = dict(
        n_top_genes=n_top_genes,
        n_neighbors=n_neighbors,
        leiden_resolution=leiden_resolution,
        filter_percentile=filter_percentile,
    )

    cell_sdatas: dict[str, spatialAdata] = {}
    if parallel:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=len(methods)) as pool:
            futures = {
                pool.submit(
                    _cluster_one_method,
                    binned_sdata_path=binned_paths[name],
                    clustered_sdata_path=clustered_paths[name],
                    **cluster_kwargs,
                ): name
                for name in methods
            }
            for fut in as_completed(futures):
                name = futures[fut]
                cell_sdatas[name] = fut.result()
                logging.info("  Finished clustering: %s", name)
    else:
        for name in methods:
            cell_sdatas[name] = _cluster_one_method(
                binned_sdata_path=binned_paths[name],
                clustered_sdata_path=clustered_paths[name],
                **cluster_kwargs,
            )
            logging.info("  Finished clustering: %s", name)

    paths_records = [
        {"method": name, "binned_sdata_path": binned_paths[name], "clustered_sdata_path": clustered_paths[name]}
        for name in methods
    ]
    paths_df = pd.DataFrame.from_records(paths_records)
    paths_df.to_csv(output_dir / "paths.csv", index=False)
    logging.info("Saved paths to %s", output_dir / "paths.csv")

    return cell_sdatas


def load_pipeline_unsupervised_results(
    output_dir: str | Path,
) -> dict[str, spatialAdata]:
    """Reload clustered sdatas saved by :func:`run_pipeline_unsupervised`."""
    output_dir = Path(output_dir)
    paths_df = pd.read_csv(output_dir / "paths.csv")
    cell_sdatas: dict[str, spatialAdata] = {}
    for _, row in paths_df.iterrows():
        logging.info("Loading %s from %s", row["method"], row["clustered_sdata_path"])
        cell_sdatas[row["method"]] = load_spatialAdata(row["clustered_sdata_path"])
    return cell_sdatas


def load_pipeline_results(
    output_dir: str | Path,
) -> dict[str, spatialAdata]:
    """
    Reload cell-typed sdatas saved by :func:`run_pipeline`.

    Reads ``paths.csv`` from *output_dir* and loads each method's
    cell-typed spatialAdata from disk.
    """
    output_dir = Path(output_dir)
    paths_df = pd.read_csv(output_dir / "paths.csv")
    cell_sdatas: dict[str, spatialAdata] = {}
    for _, row in paths_df.iterrows():
        logging.info("Loading %s from %s", row["method"], row["cell_typed_sdata_path"])
        cell_sdatas[row["method"]] = load_spatialAdata(row["cell_typed_sdata_path"])
    return cell_sdatas
