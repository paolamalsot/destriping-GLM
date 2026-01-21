from __future__ import annotations
from typing import TYPE_CHECKING
import anndata
import pandas as pd
import scipy.sparse as sp
import numpy as np
from src.utilities.df_unique_keys import coordinates_df, img_df
from src.spatialAdata.spatialAdata import spatialAdata

def make_spatial_data_from_numpy_2D(data):
    nrows, ncols = data.shape
    row_indices = np.broadcast_to(np.arange(nrows).reshape(-1,1), data.shape).flatten().reshape(-1,1)
    col_indices = np.broadcast_to(np.arange(ncols).reshape(1,-1), data.shape).flatten().reshape(-1,1)
    count_matrix = data.reshape(-1,1)
    coordinates = np.hstack([row_indices, col_indices])
    return make_sdata(coordinates, count_matrix)

def make_sdata(coordinates, count_matrix, obs = None, var = None):
    # coordinates is a n_bins x 2 numpy array
    # count_matrix is a n_bins x n_genes numpy array or a scipy sparse csr matrix

    n_counts = np.array(np.sum(count_matrix, axis = -1)).flatten() #necessary for matrix...
    if obs is None:
        obs = pd.DataFrame({"n_counts": n_counts})
    else:
        obs["n_counts"] = n_counts
    obs.index = obs.index.astype("str")

    # check count matrix is numpy
    if isinstance(count_matrix, np.ndarray):
        X = sp.csr_matrix(count_matrix)
    elif isinstance(count_matrix, sp.csr_matrix):
        X = count_matrix.copy()
    else:
        raise ValueError("count_matrix must be a numpy array or a scipy sparse matrix")
    
    library_id = "dummy_library_id"
    uns = {"spatial": {library_id: {"scalefactors": {"bin_size_um": 2.0, "microns_per_pixel": 2.0, "spot_diameter_fullres": 1.0},
                                    "images": {}}}} #doesn't have any meaning, just to have a default scalefactor of 1 !
    adata = anndata.AnnData(obs = obs, X = X, var = var, uns = uns)
    adata.obsm["coords__array"] = coordinates
    #create a spatial data with the generated_counts_matrix, and use row_idx and col_idx as "array" coordinates ! (use the correct orientation !)
    coordinates_df_ = coordinates_df.from_records(
    [{"coordinate_id": "array", 
        "img_key": pd.NA,
        "scalefactor": 1.0}]
    )
    coordinates_df_.df["img_key"] = coordinates_df_.df["img_key"].astype(object)
    img_df_ = img_df() 
    spatialdata = spatialAdata(adata, library_id = library_id, coordinate_df=coordinates_df_, img_df = img_df_)
    return spatialdata