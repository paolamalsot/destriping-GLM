from src.utilities.timestamp import add_timestamp
from src.utilities.process_10x_genomics_datasets.process import process
import os

output_dir = add_timestamp("results/Visium_HD_Zebrafish_Head_tissue/10x_segmentation__no_bin_filtering")
os.makedirs(output_dir, exist_ok=True)

path_data = "data/Visium_HD_Zebrafish_Head/binned_outputs/square_002um/"
source_img_path = (
    "data/Visium_HD_Zebrafish_Head/Visium_HD_3prime_Zebrafish_Head_tissue_image.btf"
)
path_barcodes = "data/Visium_HD_Zebrafish_Head/Visium_HD_3prime_Zebrafish_Head_barcode_mappings.parquet"

process(path_data, source_img_path, path_barcodes, output_dir, min_counts=None)
