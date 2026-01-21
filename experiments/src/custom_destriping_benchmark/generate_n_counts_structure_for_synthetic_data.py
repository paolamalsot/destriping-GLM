from src.spatialAdata.loading import load_spatialAdata
from src.spatialAdata.labels_convention import int_to_word
import scipy.sparse as sp
import datetime
from pathlib import Path
datetime_ = datetime.datetime.now().strftime("%y_%m_%d__%H:%M:%S")

data_path = "results/Visium_HD_Mouse_Brain_tissue/stardist_segmentation__24_02_25__18_08_11"
sdata = load_spatialAdata(data_path)
sdata.obs["labels_he_str"] = int_to_word(sdata.obs["labels_he"].astype(float).fillna(-1).astype(int).values)
cdata = sdata.bin2cell(labels_source_key = "labels_he_str", labels_key = "labels_he_str")
cdata.adata.X = cdata.X/cdata.obs["bin_count"].values.reshape(-1, 1)
cdata.adata.X = sp.csr_matrix(cdata.adata.X)
cdata.obs.reset_index(names = "reference_index", inplace = True)
cdata.obs.index = cdata.obs.index.astype(str)
output_dir = "results/my_notebooks/simulation_n_counts_real_data"
path = Path(output_dir) / f"cdata__{datetime_}.h5ad"
path.parent.mkdir(exist_ok = True, parents = True)
print(f"{path=}")
cdata.adata.write_h5ad(path)