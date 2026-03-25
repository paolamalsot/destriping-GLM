import json
from pathlib import Path

import pandas as pd
import yaml

from src.spatialAdata.loading import load_spatialAdata
from src.spatialAdata.spatialAdata import spatialAdata


def load_subsampled_sdatas(run_dir: str | Path, project_root: str | Path = ".") -> dict[float, spatialAdata]:
    """Load subsampled spatialAdata objects from a segmentation-sensitivity run directory.

    Reads ``.hydra/config.yaml`` to find the original sdata path and the
    cell_id_label, then for every ``pct_XX/`` sub-directory injects the
    corresponding subsampled labels into a copy of the original sdata.

    Parameters
    ----------
    run_dir:
        Path to the Hydra run output directory (the one that contains
        ``.hydra/`` and ``pct_XX/`` sub-directories).
    project_root:
        Root of the project (needed to resolve relative paths stored in the
        Hydra config).  Defaults to the current working directory.

    Returns
    -------
    dict mapping ``pct_kept`` (float, e.g. 0.95) to the corresponding
    ``spatialAdata`` with the subsampled labels in ``obs[cell_id_label]``.
    """
    run_dir = Path(run_dir)
    project_root = Path(project_root)

    config = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    path_data = project_root / config["dataset"]["path_data"]
    cell_id_label = config["dataset"]["cell_id_label"]

    data = load_spatialAdata(str(path_data))
    data.add_array_coords_to_obs()

    result = {}
    for pct_dir in sorted(run_dir.glob("pct_*")):
        metadata = json.loads((pct_dir / "metadata.json").read_text())
        pct = metadata["pct_kept"]

        labels_df = pd.read_parquet(pct_dir / "labels.parquet")
        subsampled_labels = labels_df[cell_id_label].loc[data.adata.obs_names]

        sdata = data.copy()
        sdata.adata.obs[cell_id_label] = subsampled_labels

        result[pct] = sdata

    return result


def load_split_sdatas(run_dir: str | Path, project_root: str | Path = ".") -> dict[float, spatialAdata]:
    """Load split spatialAdata objects from a splitting sensitivity run directory.

    Analogous to ``load_subsampled_sdatas`` but reads ``split_p*/`` sub-directories
    and uses the ``p_split`` metadata key.

    Returns
    -------
    dict mapping ``p_split`` (float, e.g. 0.1) to the corresponding
    ``spatialAdata`` with the split labels in ``obs[cell_id_label]``.
    """
    run_dir = Path(run_dir)
    project_root = Path(project_root)

    config = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    path_data = project_root / config["dataset"]["path_data"]
    cell_id_label = config["dataset"]["cell_id_label"]

    data = load_spatialAdata(str(path_data))
    data.add_array_coords_to_obs()

    result = {}
    for split_dir in sorted(run_dir.glob("split_p*")):
        metadata = json.loads((split_dir / "metadata.json").read_text())
        p = metadata["p_split"]

        labels_df = pd.read_parquet(split_dir / "labels.parquet")
        split_labels = labels_df[cell_id_label].loc[data.adata.obs_names]

        sdata = data.copy()
        sdata.adata.obs[cell_id_label] = split_labels

        result[p] = sdata

    return result


def load_merged_sdatas(run_dir: str | Path, project_root: str | Path = ".") -> dict[float, spatialAdata]:
    """Load merged spatialAdata objects from a merging sensitivity run directory.

    Analogous to ``load_subsampled_sdatas`` but reads ``merge_p*/`` sub-directories
    and uses the ``p_merge`` metadata key.

    Returns
    -------
    dict mapping ``p_merge`` (float, e.g. 0.1) to the corresponding
    ``spatialAdata`` with the merged labels in ``obs[cell_id_label]``.
    """
    run_dir = Path(run_dir)
    project_root = Path(project_root)

    config = yaml.safe_load((run_dir / ".hydra" / "config.yaml").read_text())
    path_data = project_root / config["dataset"]["path_data"]
    cell_id_label = config["dataset"]["cell_id_label"]

    data = load_spatialAdata(str(path_data))
    data.add_array_coords_to_obs()

    result = {}
    for merge_dir in sorted(run_dir.glob("merge_p*")):
        metadata = json.loads((merge_dir / "metadata.json").read_text())
        p = metadata["p_merge"]

        labels_df = pd.read_parquet(merge_dir / "labels.parquet")
        merged_labels = labels_df[cell_id_label].loc[data.adata.obs_names]

        sdata = data.copy()
        sdata.adata.obs[cell_id_label] = merged_labels

        result[p] = sdata

    return result
