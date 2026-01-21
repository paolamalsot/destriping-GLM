from src.utilities.timestamp import add_timestamp
from src.utilities.process_10x_genomics_datasets.process import process
import os

output_dir = add_timestamp("results/Visium_HD_Human_Lymph_Node_tissue/10x_segmentation")
os.makedirs(output_dir, exist_ok=True)

path_data = "data/Visium_HD_Human_Lymph_Node/binned_outputs/square_002um/"
source_img_path = (
    "data/Visium_HD_Human_Lymph_Node/Visium_HD_Human_Lymph_Node_FFPE_tissue_image.tif"
)

path_barcodes = "data/Visium_HD_Human_Lymph_Node/Visium_HD_Human_Lymph_Node_FFPE_barcode_mappings.parquet"

process(path_data, source_img_path, path_barcodes, output_dir)
