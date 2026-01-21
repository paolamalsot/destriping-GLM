import numpy as np
from src.spatialAdata.coordinates_df_funs import get_unscaled_coordinates
from scipy.sparse import csr_matrix
import itertools
import os
import scanpy as sc
import scipy
import warnings
from scipy.sparse import SparseEfficiencyWarning

def mpp_suffix(mpp):
    return str(mpp).replace(".", "")

def default_he_spatial_cropped_key(mpp):
    #NB: it is absurd that the mpp enters in the spatial_cropped_key since these are unscaled coordinates, but that was what bin2cell did...
    mpp_string = mpp_suffix(mpp)
    spatial_cropped_key = f"spatial_cropped_{mpp_string}"
    return spatial_cropped_key

def default_he_image_key(mpp):
    return f"{mpp}_mpp"

def get_default_scale_factor(adata, img_key, library_id):
    # returns the default scale factor to go from (unscaled) spatial coordinates (stored in obsm) to pixel coordinates of the image
    return adata.uns["spatial"][library_id]["scalefactors"][f"tissue_{img_key}_scalef"]

def replace_0_labels_by_nan(adata, labels_name):
   labels = adata.obs[labels_name]
   labels = [np.nan if x=='0' else x for x in labels]
   adata.obs[labels_name] = labels
   return adata

def scaled_he_img_path(mpp, output_dir):
    mpp_string = mpp_suffix(mpp)
    return os.path.join(output_dir, f"he_scaled_{mpp_string}.tiff")


def destripe_counts(adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted"):
    #corrected bin2cell which outputs nan when adata.obs[counts_key] = 0
    '''
    Scale each row (bin) of ``adata.X`` to have ``adjusted_counts_key`` 
    rather than ``counts_key`` total counts.
    
    Input
    -----
    adata : ``AnnData``
        2um bin VisiumHD object. Raw counts, needs to have ``counts_key`` 
        and ``adjusted_counts_key`` in ``.obs``.
    counts_key : ``str``, optional (default: ``"n_counts"``)
        Name of ``.obs`` column with raw counts per bin.
    adjusted_counts_key : ``str``, optional (default: ``"n_counts_adjusted"``)
        Name of ``.obs`` column storing the desired destriped counts per bin.
    '''
    #scanpy's utility function to make sure the anndata is not a view
    #if it is a view then weird stuff happens when you try to write to its .X
    sc._utils.view_to_actual(adata)
    #adjust the count matrix to have n_counts_adjusted sum per bin (row)
    #premultiplying by a diagonal matrix multiplies each row by a value: https://solitaryroad.com/c108.html
    bin_scaling_diag = (adata.obs[adjusted_counts_key]/adata.obs[counts_key]).values
    nans_index, = np.nonzero(adata.obs[counts_key] == 0)
    bin_scaling_diag[nans_index] = 1.0
    bin_scaling = scipy.sparse.diags(bin_scaling_diag)
    adata.X = bin_scaling.dot(adata.X)

def aggr_sum(X, indices_list):
    # X is typically a csr matrix
    # indices_list is a list of lists containing indices to aggregate
    n_out = len(indices_list)
    multiplicator = csr_matrix((n_out, X.shape[0]), dtype = bool) #by default, filled with zeros
    row_index = np.array([i for i in np.arange(n_out) for j in indices_list[i]])
    col_index = np.array(list(itertools.chain(*indices_list))) #flatten list
    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore", SparseEfficiencyWarning)  # Adjust for the specific warning type
        multiplicator[row_index, col_index] = 1
    
    out = multiplicator @ X
    return out

def aggr_mean(X, indices_list):
    # X is typically a csr matrix
    # indices_list is a list of lists containing indices to aggregate
    n_out = len(indices_list)
    counts_per_row = np.array([len(inner) for inner in indices_list])
    out = aggr_sum(X, indices_list).astype(np.float32)
    out = out/counts_per_row.reshape(-1,1)
    return out