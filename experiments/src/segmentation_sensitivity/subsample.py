import json
import os
from pathlib import Path

import numpy as np

from src.spatialAdata.loading import load_spatialAdata
from src.segmentation_sensitivity.subsampling import subsample_labels_nested


def parse_config(original_root, cfg):
    data_path = os.path.join(original_root, cfg.dataset.path_data)
    data = load_spatialAdata(data_path)
    data.add_array_coords_to_obs()

    cell_id_label = cfg.dataset.cell_id_label
    labels = data.adata.obs[cell_id_label]
    n_nuclei = int(labels.notna().sum())

    subsampled = subsample_labels_nested(
        labels=labels,
        percentages=list(cfg.percentages),
        seed=cfg.seed,
    )

    cwd = Path(os.getcwd())
    for pct, subsampled_labels in subsampled.items():
        pct_int = int(round(pct * 100))
        out_dir = cwd / f"pct_{pct_int}"
        out_dir.mkdir(parents=True, exist_ok=True)

        df = data.adata.obs[["array_row", "array_col"]].copy()
        df[cell_id_label] = subsampled_labels
        df.to_parquet(out_dir / "labels.parquet")

        n_kept = int(subsampled_labels.notna().sum())
        metadata = {
            "pct_kept": pct,
            "n_nuclei_total": n_nuclei,
            "n_nuclei_kept": n_kept,
            "seed": cfg.seed,
            "cell_id_label": cell_id_label,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"pct={pct_int}%: kept {n_kept}/{n_nuclei} nuclei → {out_dir}")
