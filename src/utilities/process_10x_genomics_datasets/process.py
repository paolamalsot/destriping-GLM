from src.spatialAdata.loading import load_visium_hd

import pandas as pd
import os

def process(path_data, source_img_path, path_barcodes, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    data = load_visium_hd(path_data, source_img_path)
    df = pd.read_parquet(
        path_barcodes
    )
    # So we have to select only those in nucleus (not cell-expansion)
    data.obs["cell_id"] = data.index.map(
        df.query("in_nucleus").set_index("square_002um")["cell_id"]
    )
    data.scale_he_image(mpp=0.5)  # for later visualization
    data.add_array_coords_to_obs()
    data.n_counts

    data.filter_genes(min_cells=3)
    data.filter_cells(min_counts=1)
    data.n_counts

    data.save(output_dir)