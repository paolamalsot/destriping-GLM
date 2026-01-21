from src.spatialAdata.loading import load_visium_hd
from src.utilities.timestamp import add_timestamp
import os
import logging

# segmented H&E
output_dir = add_timestamp("results/Visium_HD_Mouse_Brain_tissue/stardist_segmentation")
os.makedirs(output_dir, exist_ok=True)

path_data = "data/Visium_HD_Mouse_Brain/binned_outputs/square_002um/"
source_img_path = "data/Visium_HD_Mouse_Brain/Visium_HD_Mouse_Brain_tissue_image.tif"
data = load_visium_hd(path_data, source_img_path)
min_1_coord = data.get_unscaled_coordinates("array")[:,1].min()
logging.info(f"Minimum coordinate value in the second dimension: {min_1_coord}")
shift = 486
logging.info(f"Shifting vertical coordinate by {shift}")
# I think this is not necessary, but it is to ensure consistency with previous outputs and the new load_visium_hd
data.obsm["coords__array"][:, 1] += shift
data.add_array_coords_to_obs()
min_1_coord = data.get_unscaled_coordinates("array")[:,1].min()
logging.info(f"Minimum coordinate value in the second dimension after shift: {min_1_coord}")
#filtering data
data.filter_genes(min_cells=3)
data.filter_cells(min_counts=1)

#cell-segmentation
data.segment_he_stardist(mpp = 0.5, output_dir=output_dir)
data.save(output_dir)