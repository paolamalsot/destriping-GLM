import json
import os
from pathlib import Path

import numpy as np

from src.spatialAdata.loading import load_spatialAdata
from src.segmentation_sensitivity.splitting import split_cells_nested


def parse_config(original_root, cfg):
    data_path = os.path.join(original_root, cfg.dataset.path_data)
    data = load_spatialAdata(data_path)
    data.add_array_coords_to_obs()

    cell_id_label = cfg.dataset.cell_id_label
    df = data.adata.obs[["array_row", "array_col", cell_id_label]].copy()
    n_nuclei_before = int(df[cell_id_label].nunique())

    rng = np.random.default_rng(cfg.seed)
    split_results = split_cells_nested(
        df=df,
        cell_id_label=cell_id_label,
        ps=list(cfg.ps),
        rng=rng,
    )

    cwd = Path(os.getcwd())
    for p, split_labels in split_results.items():
        p_int = int(round(p * 100))
        out_dir = cwd / f"split_p{p_int}"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_df = data.adata.obs[["array_row", "array_col"]].copy()
        out_df[cell_id_label] = split_labels
        out_df.to_parquet(out_dir / "labels.parquet")

        n_nuclei_after = int(split_labels.nunique())
        metadata = {
            "p_split": p,
            "n_nuclei_before": n_nuclei_before,
            "n_nuclei_after": n_nuclei_after,
            "seed": cfg.seed,
            "cell_id_label": cell_id_label,
        }
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"p_split={p_int}%: {n_nuclei_before} → {n_nuclei_after} nuclei → {out_dir}")
