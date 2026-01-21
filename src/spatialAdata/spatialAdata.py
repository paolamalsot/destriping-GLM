from __future__ import annotations
from typing import TYPE_CHECKING

import matplotlib.axes
import matplotlib.image
if TYPE_CHECKING:
    from src.spatialAdata.spatialAdata import spatialAdata
import anndata
import pandas as pd
import logging
import matplotlib
from copy import deepcopy
import src.utilities.adata_utilities as adata_utilities
import os
import numpy as np
from src.utilities.preserve_warnings import preserve_warnings
with preserve_warnings(): #-> otherwise warning formats are changed (in stardist/__init__.py)
    import bin2cell as b2c
from bin2cell import mpp_to_scalef
from src.utilities.scanpy_spatial_squares import spatial
import scanpy as sc
from src.utilities.bin2cell_utilities import *
from src.utilities.utilities import *
import src.utilities.cv2_utils as cv2_utils
from src.utilities.cv2_utils import cv2
from src.spatialAdata.coordinates_df_funs import *
import src.utilities.sdata_utilities as sdata_utilities
from src.utilities.sdata_utilities import history_decorator, is_rgb, imshow
from src.utilities.df_unique_keys import coordinates_df, img_df
import matplotlib.pyplot as plt
from src.utilities.quantile_matching import quantile_match_sparse, quantile_match_poisson, get_qm_fun
from src.utilities.custom_imshow import custom_imshow
from src.utilities.adata_utilities import img_2D_to_vals
from numpy.typing import NDArray
from anndata import AnnData
import skimage
import json

import numpy as np
import scipy.sparse as sp

def match_sparse_type(input, reference):
    """
    Convert the input matrix to match the type of the reference matrix.
    
    If the reference is a SciPy sparse matrix, the input will be converted to a match the sparse type of the matrix.
    If the reference is a NumPy array, the input will be converted to a dense NumPy array.
    
    Parameters:
    - input: The matrix to convert (can be NumPy or SciPy sparse).
    - reference: The reference matrix (either NumPy or SciPy sparse).
    
    Returns:
    - Converted input matrix to match the type of reference.
    """
    # If reference is a SciPy sparse matrix
    if sp.issparse(reference):
        # Convert input to match sparse type of matrix
        return reference.__class__(input)

    # If reference is a NumPy array
    elif isinstance(reference, np.ndarray):
        # Convert input to dense NumPy array if it's sparse
        if sp.issparse(input):
            return input.toarray()
        else:
            return input # Ensure it's a NumPy array
    
    else:
        raise ValueError("Reference matrix must be either a NumPy array or a SciPy sparse matrix.")

def downscaled_he_img_key(mpp):
        return str(mpp) + "_mpp"


class spatialAdata:

    def __init__(
        self,
        adata: AnnData,
        library_id: str,
        coordinate_df: coordinates_df,
        img_df: img_df,
        history=None,
    ):

        # attributes
        self.adata = adata
        self.library_id = library_id
        self.coordinate_df = coordinate_df
        self.img_df = img_df

        # history -> history of what happened to this object in terms of preprocessing.
        if history is None:
            self.history = []
        else:
            self.history = history

    ## UTILITIES METHODS

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.adata.write(os.path.join(output_dir, "adata.h5ad"))
        self.coordinate_df.save(os.path.join(output_dir, "coordinate_df.csv"))
        self.img_df.save(os.path.join(output_dir, "img_df.csv"))

        with open(
            os.path.join(output_dir, "spatial_adata_attributes.json"), "w"
        ) as file:
            json.dump(
                {"library_id": self.library_id, "history": self.history}, file, indent=4
            )
            os.makedirs(output_dir, exist_ok=True)

    def copy(self):
        adata_copy = self.adata.copy()
        history_copy = deepcopy(self.history)
        coordinate_df_copy = self.coordinate_df.copy()
        img_df_copy = self.img_df.copy()
        return spatialAdata(
            adata_copy, self.library_id, coordinate_df_copy, img_df_copy, history_copy
        )

    def show_adata(self):
        show_adata(self.adata)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n\nadata:\n{repr(self.adata)}\ncoordinate_df:\n{repr(self.coordinate_df)}\nimg_df:\n{repr(self.img_df)}\nhistory:\n{self.history}\nlibrary_id: {self.library_id})"

    # indexing
    def __getitem__(self, index):
        """Returns a sliced view of the object."""

        adata = adata_utilities.get_item(self.adata, index)        
        # Construct the new sliced object as before
        sdata = spatialAdata(
            adata, self.library_id, self.coordinate_df, self.img_df, self.history
        )

        return sdata     

    @history_decorator
    def zoom(self, limits, coordinate_id="array", copy=True):

        # returns a spatialAdata object with a view of adata
        # limits is an array wrt coordinate_id, with [min(coordinates[:,0]), max(coordinates[:,0]), min(coordinates[:,1]), max(coordinates[:,1])]

        if copy:
            return self.copy().zoom(limits, coordinate_id, copy=False).copy() #not super elegant solution...

        coordinates = self.get_unscaled_coordinates(coordinate_id=coordinate_id)
        limits_selector = selector_within_limits(coordinates, limits)
        self.adata = self.adata[limits_selector]

        return self

    ## PROCESSING METHODS

    @history_decorator
    def scale_he_image(self, mpp=1, crop=True, buffer=150, img_path=None):

        # scales the fullres he image to the desired mpp.
        # If crop, we reduce the image to the area that contains the count data. When specified buffer pixels from the original image around the crop area are conserved.
        # img_path: where to save the resulting image !

        img = self.get_img("fullres")

        if crop:
            img, new_coordinates = sdata_utilities.crop(
                img, self.get_coordinates("fullres", round = False, truncate= False), buffer
            )
            coordinate_id = "fullres_cropped"
            self.set_unscaled_coordinates(coordinate_id, new_coordinates)
            # NB: we don't store this cropped version of the fullres img
        else:
            coordinate_id = "fullres"

        # reshape image to desired microns per pixel
        # get necessary scale factor for the custom mpp
        # multiply dimensions by this to get the shrunken image size

        scalef = mpp_to_scalef(self.adata, mpp=mpp)
        # need to reverse dimension order and turn to int for cv2
        # TODO: looks super nasty !!! Is it true ??
        dim = (np.array(img.shape[:2]) * scalef).astype(int)[::-1]
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img_key = downscaled_he_img_key(mpp)

        self.add_img(
            img, img_key, img_path, scalefactor=scalef, in_memory=True, rgb=True
        )

        # TODO: can we remove that ? Initially we did it to have compatibility with bin2cell
        self.adata.uns["spatial"][self.library_id]["scalefactors"][
            "tissue_" + str(mpp) + "_mpp_scalef"
        ] = scalef

        self.add_coordinates(
            coordinate_id=coordinate_id, scalefactor=scalef, img_key=img_key
        )

    @history_decorator
    def segment_he_stardist(self, mpp, output_dir, prob_thresh=0.01):
        # inspired from bin2cell segment_stardist
        """
        Scaling image

        Performed by bin2cell.scaled_he_image
        In short: scales the H&E image.

        1. stores the cropped scaled image. NB: the image is cropped to fit the spatial coordinates. Note image is stored up-down, left-right.
            Stored in adata.uns['spatial'][library]["images"][x_mpp]
        2. stores the coordinates wrt to unscaled *cropped* image. spatial[:,1] is up-down, spatial[:,0] is left-right
            Stored in adata.obsm[spatial_cropped_key]
        3. stores the scale_factor to go from unscaled *cropped* coordinates (stored in (2.)) to the pixel coordinates of the image (stored in (1.))
            Stored in adata.uns['spatial'][library]['scalefactors']['tissue_'+str(mpp)+"_mpp_scalef"]

            Calculation of the scale factor:
            Let $x$ represent the coordinate with respect to the image, $n$ its number of pixels, and mpp, its resolution in microns-per-pixel. (The subscript s refers to the scaled image, for example $x_s$ refers to the pixel coordinate wrt scaled image)

            $f_s = x_s/x = mpp/mpp_s$

        NB: library = list(adata.uns['spatial'].keys())[0] by default in bin2cell.scaled_he_image

        ### Running stardist on the scaled image
        Segments the nuclei from the scaled H&E image.

        ### Insertion of the labels
        Adds nuclei labels to the anndata object (under adata.obs['labels_he'])
        """
        print("new_method")
        os.makedirs(output_dir, exist_ok=True)

        img_path = scaled_he_img_path(mpp, output_dir)
        self.scale_he_image(mpp, img_path=img_path)
        mpp_string = mpp_suffix(mpp)

        labels_npz_path = os.path.join(output_dir, f"he_scaled_{mpp_string}.npz")

        b2c.stardist(
            image_path=img_path,
            labels_npz_path=labels_npz_path,
            stardist_model="2D_versatile_he",
            prob_thresh=prob_thresh,
        )

        self.insert_labels(
            labels_npz_path=labels_npz_path,  # file with pixel coordinates and labels of cells...
            img_key=self.downscaled_he_img_key(mpp),  # was it based on H&E (spatial) or GEX
        )

        self.save(os.path.join(output_dir, "sdata"))

        return self

    @history_decorator
    def segment_gex_stardist(self, mpp, output_dir):
        self.n_counts
        # in bin2cell they use n_counts_adjusted...
        # wierd that they do not use log1p
        os.makedirs(output_dir, exist_ok=True)
        img_path = os.path.join(output_dir, "gex.tiff")
        img_key = "gex_stardist"
        labels_key = "labels_gex"
        self.n_counts
        if "n_counts_adjusted" in self.obs.columns:
            assert np.all(np.isclose(self.obs["n_counts"], self.obs["n_counts_adjusted"]))
        self.grid_image_from_label_custom_mpp("n_counts", new_image_label = img_key, log1p = False, mpp = mpp, sigma = 5, save_path=img_path, pad_to = (3350, 3350))

        labels_path = os.path.join(output_dir, "gex.npz")
        b2c.stardist(image_path=img_path, 
             labels_npz_path=labels_path, 
             stardist_model="2D_versatile_fluo", 
             prob_thresh=0.05, 
             nms_thresh=0.5
            )

        self.insert_labels(
            labels_npz_path=labels_path,  # file with pixel coordinates and labels of cells...
            img_key=img_key,  # was it based on H&E (spatial) or GEX
            labels_key=labels_key
        )

        self.save(os.path.join(output_dir, "sdata"))


    def insert_labels(self, labels_npz_path, img_key, labels_key="labels_he"):
        # inspired by bin2cell

        # load sparse segmentation results
        labels_sparse = scipy.sparse.load_npz(labels_npz_path)

        coords = self.get_coordinates(img_key)

        # original comment in bin2cell: there is a possibility that some coordinates will fall outside labels_sparse
        # For example, in the crc data, the microscope image does not cover the full array, so some coordinates are outside the image.

        # start by pregenerating an obs column of all zeroes so all bins are covered
        self.adata.obs[labels_key] = 0
        # can now construct a mask defining which coordinates fall within range
        # apply the mask to the coords and the obs to just go for the relevant bins
        mask = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < labels_sparse.shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < labels_sparse.shape[1])
        )

        assert np.all(mask == self.within_img(img_key))

        # pull out the cell labels for the coordinates, can just index the sparse matrix with them
        # insert into bin object, need to turn it into a 1d numpy array from a 1d numpy matrix first
        self.adata.obs.loc[mask, labels_key] = np.asarray(
            labels_sparse[coords[mask, 0], coords[mask, 1]]
        ).flatten()

        # convert the int to categorical dtype with strings
        NA_positions = (self.adata.obs[labels_key] == 0)
        self.adata.obs[labels_key] = self.adata.obs[labels_key].astype(str)
        self.adata.obs.loc[NA_positions, labels_key] = pd.NA
        self.adata.obs[labels_key] = self.adata.obs[labels_key].astype("category")
        # replace_0_labels_by_nan(self.adata, labels_key)

    @history_decorator
    def destripe_bin2cell(self):
        original_counts = self.n_counts
        b2c.destripe(
            self.adata,
            adjust_counts=False
        )  # causes the warning FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`

        self.n_counts
        n_counts_adjusted = self.adata.obs["n_counts_adjusted"].copy()

        # not done in bin2cell keep inf and Nan values in n_counts_adjusted the same !

        n_counts_adjusted.loc[pd.isna(n_counts_adjusted)] = original_counts[
            pd.isna(n_counts_adjusted)
        ]
        n_counts_adjusted.loc[np.isinf(n_counts_adjusted)] = original_counts[
            np.isinf(n_counts_adjusted)
        ]
        self.adata.obs["n_counts_adjusted"] = n_counts_adjusted

        destripe_counts(
            self.adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted"
        )
        self.n_counts

        return self
    
    @history_decorator
    def bin2cell(self, labels_key = "labels_he", labels_source_key = "labels_he_source", diameter_scale_factor=None, aggr_fun = aggr_sum):
        # Return a new spatial data object which aggregates all bins with same labels
        # Gene expression will be added up, coordinates averaged.
        # ``"spot_diameter_fullres"`` in the scale factors multiplied by
        # ``diameter_scale_factor`` to reflect increased unit size. Returns cell level AnnData,
        # including ``.obs["bin_count"]`` reporting how many bins went into creating the cell.

        def aggregate_fun(group_df):
            dict_ = {#labels_key: group_df[labels_key], #will that work, since labels_key is the groupby ??
                     "bin_count": len(group_df),
                     "original_indices": group_df["num_index"].tolist(), #will that work, since we are doing a groupby ??
                     }
            if not(labels_source_key is None):
                labels_source_unique = np.unique(group_df.loc[:,labels_source_key]).tolist()
                assert len(labels_source_unique) == 1
                dict_[labels_source_key] = labels_source_unique[0]

            out_series = pd.Series(dict_)
            return out_series

        self.adata.obs["num_index"] = np.arange(len(self.adata.obs))
        b2c_obs = self.adata.obs.dropna(axis = 0, subset = labels_key).groupby(labels_key, observed = True).apply(aggregate_fun)
        del self.adata.obs["num_index"]
        b2c_obs.index = b2c_obs.index.astype(str) #avoids when loading aligned_df.py (67): Transforming to str index.
        # b2c_obs[labels_key] = b2c_obs.index.astype(str)
        # b2c_obs.reset_index(inplace = True)
        # b2c_obs[labels_key] = b2c_obs.index.astype("category")
        # create the new cdata object
        b2cdata = self[:len(b2c_obs)].copy() #random selection of indices, just to get the correct size.
        b2cdata.adata.obs = b2c_obs

        # adding the gene expression data
        indices_aggr = b2c_obs["original_indices"].tolist()
        del b2c_obs["original_indices"]
        b2c_expr = aggr_fun(self.adata.X, indices_aggr)
        b2cdata.adata.X = b2c_expr

        if diameter_scale_factor is None:
            diameter_scale_factor = np.sqrt(np.mean(b2c_obs.bin_count))
        try:
            b2cdata.adata.uns['spatial'][b2cdata.library_id]['scalefactors']['spot_diameter_fullres'] *= diameter_scale_factor
        except:
            logging.info("No spot diameter")

        # mean of the spatial coordinates
        for coord in np.unique(self.coordinate_df.df["coordinate_id"]): #TODO: we could implement .loc in coordinate_df
            coordinates = self.get_unscaled_coordinates(coord)
            new_coordinates = aggr_mean(coordinates, indices_aggr)
            b2cdata.set_unscaled_coordinates(coord, new_coordinates)

        # I did not do the thing with source - according to me, does not make sense, since the source can vary...

        return b2cdata

    @history_decorator
    def bin_level_normalization(self, quantile, nucl_only, cell_id_label):
        # adjusts the total counts of each cell to be equal to the xth quantile of tot-counts
        # if nucl_only: calculates the quantile on nuclear counts only
        self.n_counts
        if nucl_only:
            subset = self.obs.loc[self.nucl_indices(cell_id_label)]
        else:
            subset = self.obs
        target_sum = subset["n_counts"].quantile(quantile)
        self.adata.obs["n_counts_adjusted"] = target_sum
        self.normalize_total(target_sum)
        self.n_counts

    @history_decorator
    def get_destripe_factors(self, row_factors: pd.Series, col_factors: pd.Series):
        array_coords = self.get_unscaled_coordinates("array")
        array_row = array_coords[:,0]
        array_col = array_coords[:,1]

        # row_factors_filled = pd.Series(index=np.unique(array_row), data = np.ones(len(np.unique(array_row))))
        # row_factors_index_intersect = row_factors_filled.index.intersection(row_factors.index)
        # row_factors_filled.loc[row_factors_index_intersect] = row_factors.loc[row_factors_index_intersect]

        row_factors_filled = row_factors.reindex(np.unique(array_row), fill_value=1.0)
        col_factors_filled = col_factors.reindex(np.unique(array_col), fill_value=1.0)
        # col_factors_filled = pd.Series(index=np.unique(array_col), data = np.ones(len(np.unique(array_col))))
        # col_factors_index_intersect = col_factors_filled.index.intersection(col_factors.index)
        # col_factors_filled.loc[col_factors_index_intersect] = col_factors.loc[col_factors_index_intersect]

        row_factors_ = row_factors_filled.loc[array_row].values
        col_factors_ = col_factors_filled.loc[array_col].values

        return (row_factors_ * col_factors_)

    @history_decorator
    def destripe_dividing_factors(self, row_factors: pd.Series, col_factors: pd.Series):
        self.n_counts
        self.obs["destripe_factor"] = self.get_destripe_factors(row_factors, col_factors)
        self.obs["n_counts_adjusted"] = self.obs["n_counts"]
        self.obs["n_counts_adjusted"] = self.obs["n_counts_adjusted"]/self.obs["destripe_factor"]
        # correcting the n_counts_ajusted which are inf and nans caused by n_counts = 0 and destripe_factor = 0...
        inf_selector = np.isinf(self.obs["n_counts_adjusted"])
        self.obs.loc[inf_selector, "n_counts_adjusted"] = self.obs.loc[inf_selector, "n_counts"]
        zero_zero_selector = (self.obs["n_counts"].values == 0) & (self.obs["destripe_factor"].values == 0)
        self.obs.loc[zero_zero_selector, "n_counts_adjusted"] = 0
        assert not (self.obs.loc[zero_zero_selector, "n_counts_adjusted"].isna().any())
        destripe_counts(self.adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted")
        self.n_counts
        return self

    @history_decorator
    def destripe_quantile_matching(self, row_factors: pd.Series, col_factors: pd.Series, gene_expression_per_bin : AnnData, method = quantile_match_poisson):

        # gene_expression_per_bin is an AnnData object with indices corresponding self.index
        self.n_counts

        # select portion of obs to destripe
        self.obs["destripe_factor"] = self.get_destripe_factors(row_factors, col_factors)

        # if gene_expression_per_cell is None:
        #     gene_expression_per_cell = self.expected_cell_ge_conc_from_stripe_factors(row_factors, col_factors)

        # list_cell_ids = gene_expression_per_cell.obs_names #NB: .obs_names is the alias for .obs.index
        # bin_to_correct_indices = self.obs.query("@cell_id in @list_cell_ids").index
        # bin_to_correct_cell_id = self.obs.query("@cell_id in @list_cell_ids")[cell_id_label].values

        logging.debug("1")
        # expression = self.X.copy() #TODO: removed this ?? unnecessary I think...
        expression = self.X
        logging.debug("2")
        expected_concentration = gene_expression_per_bin[self.adata.obs_names].X
        logging.debug("3")
        destripe_factors = self.obs["destripe_factor"].values
        logging.debug("4")
        logging.debug(expression)
        logging.debug(expected_concentration)
        corrected_expression = quantile_match_sparse(expression, expected_concentration, destripe_factors.reshape(-1,1), method = method)
        logging.debug("5")
        # assign corrected expression and correct total counts
        # int_indices_correction = self.obs.index.get_indexer(bin_to_correct_indices)
        logging.debug("6")
        if self.X.dtype == np.int64: # works both for numpy and scipy sparse !
            self.adata.X = self.adata.X.astype(np.float64) # This is important, otherwise the next line will cast to an int...
        logging.debug("7")
        # assign_view(self.X, match_sparse_type(corrected_expression, self.X)) #TODO: check this works when X is CSR/COO
        self.adata.X = match_sparse_type(corrected_expression, self.X)
        logging.debug("8")
        self.n_counts
        logging.debug("9")
        return self

    @history_decorator
    def destripe_dividing_factors_qm_tot_counts(self, row_factors: pd.Series, col_factors: pd.Series, gene_expression_per_bin : AnnData, dist = "poisson", dist_params = None):
        self.n_counts
        method = get_qm_fun(dist, dist_params=dist_params)

        self.obs["destripe_factor"] = self.get_destripe_factors(row_factors, col_factors)

        # if gene_expression_per_cell is None:
        #     gene_expression_per_cell = self.expected_cell_ge_conc_from_stripe_factors(row_factors, col_factors, cell_id = cell_id_label)

        gene_expression_per_bin.obs["n_counts"] = gene_expression_per_bin.X.sum(-1)
        self.obs["n_counts_adjusted"] = method(self.obs.n_counts.values,
                                                       gene_expression_per_bin.obs.n_counts.loc[self.adata.obs_names].values,#normally no need to put .astype str! 
                                                       self.obs["destripe_factor"].values)
        destripe_counts(self.adata, counts_key="n_counts", adjusted_counts_key="n_counts_adjusted")
        self.n_counts
        return self

    @history_decorator
    def destripe_combi_nucl_cyto(self, nucl_indices : pd.Index, method_nucl = destripe_dividing_factors_qm_tot_counts, method_cyto = destripe_dividing_factors, args_nucl = None, args_cyto = None):

        if args_cyto is None:
            args_cyto = {}
        if args_nucl is None:
            args_nucl = {}

        cyto_indices = self.obs.index.difference(nucl_indices)
        # cyto_indices = self.obs[pd.isna(self.obs[cell_id_label])].index
        # nucl_indices = self.obs[~pd.isna(self.obs[cell_id_label])].index
        cyto = self[cyto_indices].copy()
        nucl = self[nucl_indices].copy()
        # args_cyto = match_method_signature(method_cyto, args_cyto)
        # args_nucl = match_method_signature(method_nucl, args_nucl)
        method_cyto(cyto, **args_cyto)
        method_nucl(nucl, **args_nucl)
        new_adata = anndata.concat([cyto.adata, nucl.adata], uns_merge = "same")
        new_adata.uns = deepcopy(self.adata.uns)
        new_adata = new_adata[self.adata.obs_names] #aligning
        new_adata = new_adata.copy() #avoids having a view...
        self.adata = new_adata
        return self

    @history_decorator
    def round_gene_expression(self):
        self.adata.X.data = np.round(self.adata.X.data)
        return self

    @history_decorator
    def filter_genes(self, min_cells):
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        return self

    @history_decorator
    def filter_cells(self, min_counts = None, min_genes = None):
        sc.pp.filter_cells(self.adata, min_counts=min_counts, min_genes = min_genes)
        return self

    @history_decorator
    def highly_variable_genes(self, n_top_genes = None, flavor = None):
        sc.pp.highly_variable_genes(self.adata,n_top_genes=n_top_genes,flavor=flavor)

    @history_decorator
    def normalize_total(self, target_sum):
        sc.pp.normalize_total(self.adata,target_sum=target_sum)

    @history_decorator
    def log1p(self):
        sc.pp.log1p(self.adata)

    @history_decorator
    def calculate_qc_metrics(self):
        sc.pp.calculate_qc_metrics(self.adata,inplace=True)

    @history_decorator
    def preprocess(self, destripe, log1p, highly_variable, scale, copy=True):

        if copy:
            spatial_data = self.copy()
            return spatial_data.preprocess(
                destripe, log1p, highly_variable, scale, copy=False
            )

        # destripe
        if destripe:
            self.destripe_bin2cell()
        else:
            self.n_counts

        if log1p:
            sc.pp.log1p(self.adata, copy=False)

        if highly_variable:
            sc.pp.highly_variable_genes(self.adata, flavor="seurat", n_top_genes=1000)

        if scale:
            sc.pp.scale(self.adata, zero_center=False)

        return self

    ### PLOTTING METHODS

    def plot_sc_spatial(self, args):
        # the goal of this function is to adapt the coordinates to the convention in adata : origin top left and first coordinate is the horizontal !

        basis = args["basis"][8:]
        temp = self.get_unscaled_coordinates(basis)
        new = np.flip(temp.copy(), axis=-1)
        self.set_unscaled_coordinates(basis, new)
        crop_coord = [
            np.min(new[:, 0]),
            np.max(new[:, 0])+1,
            np.min(new[:, 1]),
            np.max(new[:, 1])+1,
        ]
        res = spatial(**args, crop_coord=crop_coord)

        self.set_unscaled_coordinates(basis, temp)

        return res

    def plot_obs(self, obs_name, img_key, **kwargs):
        args = self.default_sc_plotting_args(obs_name, img_key)
        # overwrite the default arguments with kwargs
        args = {**args, **kwargs}
        res = self.plot_sc_spatial(args)
        return res

    def plot_labels(self, img_key, label = "labels_he"):
        return self.plot_obs([None, label], img_key)

    def plot_shell_numbers(self, img_key):
        return self.plot_obs([None, "shell_number"], img_key)

    def plot_closest_nucl(self, img_key, filter_nuclei=True):

        # important to run after show_labels_adata #TODO: I have the impression that one needs to show it somehow ?? Like that if one does return_fig = True, with show = False, it won't work...
        # filter_nuclei: remove bins whose closest nucleus falls outside the anndata

        palette = get_palette_from_labels_he_colors(self.adata)

        if filter_nuclei:
            he_labels = self.adata.obs["labels_he"].tolist()
            sdata_pl = self[self.adata.obs["closest_nucl_label"].isin(he_labels)].copy()
        else:
            sdata_pl = self

        args = sdata_pl.default_sc_plotting_args([None, "closest_nucl_label"], img_key)
        # replace default arguments
        args["palette"] = palette

        return sdata_pl.plot_sc_spatial(args)

    def plot_img_region(self, unscaled_limits, coordinate_id, img_key, ax=None):

        img = self.get_img(img_key)
        limits_pixel_coords = get_limits_pixel_coordinates(
            img_key, coordinate_id, unscaled_limits, self.coordinate_df, self
        )

        if ax is None:
            fig, ax = plt.subplots()

        # full image
        img_selection = img[
            limits_pixel_coords[0] : limits_pixel_coords[1] + 1,
            limits_pixel_coords[2] : limits_pixel_coords[3] + 1,
        ]
        ax = imshow(img_selection, ax)
        ax.set_xticks([])
        ax.set_yticks([])

        return ax

    def plot_img_region_inset(self, unscaled_limits, coordinate_id, img_key, ax=None):

        if ax is None:
            fig, ax = plt.subplots()

        img = self.get_img(img_key)
        limits_pixel_coords = get_limits_pixel_coordinates(
            img_key, coordinate_id, unscaled_limits, self.coordinate_df, self
        )

        # full image
        ax = imshow(img, ax)
        ax.set_xticks([])
        ax.set_yticks([])

        axins = ax.inset_axes(
            [1.1, 0, 1.2, 1.2],
            xlim=limits_pixel_coords[2:],
            ylim=limits_pixel_coords[0:2],
            xticks=[],
            yticks=[],
        )

        axins = imshow(img, axins)

        ax.indicate_inset_zoom(axins, edgecolor="black", linestyle = "--")
        axins.invert_yaxis()

        plt.tight_layout()
        return ax

    def plot_n_counts_region(self, unscaled_limits, coordinate_id, img_key=None):
        # plots side by side the H&E and the counts image

        if img_key is None:
            img_key = self.downscaled_he_img_key()

        fig, axes = plt.subplots(1, 2)
        he_ax = axes[0]
        self.plot_img_region(unscaled_limits, coordinate_id, img_key, ax=he_ax)
        n_counts_ax = axes[1]
        self.plot_img_region(
            unscaled_limits, coordinate_id, "log_n_counts", ax=n_counts_ax
        )
        return axes

    def plot_n_counts(self, axis = None, colorbar_loc='right', vmax=None, vmin=None, label_counts = "n_counts", **kwargs):
        # NB: colorbar loc None => no colorbar is added.
        if axis is None:
            fig, axis = plt.subplots()
        axis.patch.set_facecolor("black")
        kwargs_updated = {"img_key": None,
                  "ax": axis,
                  "color_map": "gray",
                  "colorbar_loc": colorbar_loc,
                  "vmax": vmax,
                  "vmin": vmin,
                  "facecolor": "m"}
        kwargs_updated = {**kwargs_updated, **kwargs}
        self.plot_obs(label_counts, **kwargs_updated) #add a black and white !
        return axis

    @history_decorator
    def matrix_from_label(self, obs_label="n_counts"):
        vals = pull_values_adata(obs_label, self.adata)
        return img_2D_from_vals(self.array_coords, vals)

    def imshow_matrix_from_label(self, obs_name, axis: matplotlib.axes.Axes | None = None, cmap = "gray", na_color = "red", colorbar_on = True, **kwargs) -> matplotlib.image.AxesImage:
        matrix = self.matrix_from_label(obs_name)
        return custom_imshow(matrix, axis, cmap, na_color, colorbar_on, **kwargs)

    def label_from_matrix(self, matrix, new_label):
        vals = img_2D_to_vals(self.array_coords, matrix)
        self.obs[new_label] = vals

    @history_decorator
    def grid_image_from_label(
        self, obs_label="n_counts", new_image_label=None, log1p=True, save_path=None
    ):
        """
        Modification of bin2cell.grid_image
        Got rid of this array_check, which is more confusing than anything !!!
        Also save somewhere the colormap ?
        """

        img = self.matrix_from_label(obs_label)

        cv2_utils.save_tif_img_with_colorbar(img, save_path, log1p)

        if new_image_label is None:
            new_image_label = obs_label + ("__log1p" if log1p else "")

        scalefactors_dict = self.adata.uns["spatial"][self.library_id]["scalefactors"]
        img_scalefactor = 1.0 / (
            (scalefactors_dict["bin_size_um"] / scalefactors_dict["microns_per_pixel"])
        )

        self.add_img(
            img,
            new_image_label,
            None,
            scalefactor=img_scalefactor,
            in_memory=True,
            rgb=False,
        )

        self.add_coordinates(
            coordinate_id="array", scalefactor=1.0, img_key=new_image_label
        )

    def plot_n_counts_labels_superposition(self,axes = None, cell_id_label = "labels_he"):
        if axes is None:
            fig, axes = plt.subplots(1,3, figsize = (15,5))
        ax = self.plot_n_counts(axis = axes[0])
        ax = self[~pd.isna(self.obs[cell_id_label])].plot_obs(
            cell_id_label,
            alpha=0.4,
            ax=axes[0],
            img_key=None,
            legend_loc=None,
            na_color=(0, 0, 0, 0),
        )
        ax = self.plot_n_counts(axis = axes[1])
        ax = self[~pd.isna(self.obs[cell_id_label])].plot_obs(
            cell_id_label,
            alpha=0.4,
            ax=axes[2],
            img_key=None,
            legend_loc=None,
            na_color=(0, 0, 0, 0),
        )
        # ax = zoomed_data[~pd.isna(zoomed_data.obs.labels_he)].plot_obs("labels_he", alpha = 0.4, ax = axes[2], img_key = None, legend_loc = None, na_color = (0,0,0,0))
        # NB: bug in sp spatial, when alpha is specified, na_color takes the same alpha...
        plt.tight_layout()
        return axes

    @history_decorator
    # on the long term, this could replace grid_image_from_label
    def grid_image_from_label_custom_mpp(
        self, obs_label="n_counts", new_image_label=None, log1p=True, mpp = 0.5, sigma = 5, save_path=None, pad_to = None
    ):
        """
        Modification of bin2cell.grid_image
        Got rid of this array_check, which is more confusing than anything !!!
        Also save somewhere the colormap ?
        """

        vals = pull_values_adata(obs_label, self.adata)

        # make the values span from 0 to 255
        vals = (255 * (vals - np.min(vals)) / (np.max(vals) - np.min(vals))).astype(
            np.uint8
        )

        # optionally log1p
        if log1p:
            vals = np.log1p(vals)
            vals = (255 * (vals - np.min(vals)) / (np.max(vals) - np.min(vals))).astype(
                np.uint8
            )

        # can now create an empty image the shape of the grid and stick the values in based on the coordinates
        max_0 = np.max(self.array_coords[:, 0])
        max_1 = np.max(self.array_coords[:, 1])

        if not(pad_to is None):
            shape = (max(max_0 + 1, pad_to[0]), max(max_1 + 1, pad_to[1]))
        else:
            shape = (max_0 + 1, max_1 + 1)

        img = np.zeros(shape, dtype=np.uint8)

        img[self.array_coords[:, 0], self.array_coords[:, 1]] = vals

        scalefactors_dict = self.adata.uns["spatial"][self.library_id]["scalefactors"]
        bin_size_um = scalefactors_dict["bin_size_um"]

        scalefactor = bin_size_um/mpp

        if mpp != 2:
            dim = np.round(np.array(img.shape) * scalefactor).astype(int)[::-1]
            img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
        # run through the gaussian filter if need be
        if sigma is not None:
            img = skimage.filters.gaussian(img, sigma=sigma)
            img = (255 * (img-np.min(img))/(np.max(img)-np.min(img))).astype(np.uint8)

        # save or return image
        if save_path is not None:
            cv2_utils.save_img(img, save_path, rgb=False)

        if new_image_label is None:
            new_image_label = obs_label + ("__log1p" if log1p else "")

        img_scalefactor = 1.0 / (
            (mpp / scalefactors_dict["microns_per_pixel"])
        )

        self.add_img(
            img,
            new_image_label,
            save_path,
            scalefactor=img_scalefactor,
            in_memory=True,
            rgb=False,
        )

        self.add_coordinates(
            coordinate_id="array", scalefactor=scalefactor, img_key=new_image_label
        )

    ### PLOTTING UTILITIES

    def default_sc_plotting_args(self, obs_name, img_key):
        # select a coordinate_id which is compatible with img_key, and take the correct scale-factor
        coordinate_id, scale_factor = self.coordinates_sf_for_image(img_key)

        if img_key is not None:
            img = self.get_img(img_key)

            if not (is_rgb(img)):
                img = np.reshape(img, (*img.shape, 1)).repeat(3, axis=-1)
                bw = True
            else:
                bw = False

            spotsize = self.spotsize * self.get_img_scalefactor(img_key)
            spotsize = spotsize / scale_factor  # by definition of sc.spatial
        else:
            bw = False
            img = None
            spotsize = 1

        args = {
            "adata": self.adata,
            "bw": bw,
            "img": img,
            "basis": f"coords__{coordinate_id}",
            "scale_factor": scale_factor,
            "na_color": na_color,
            "return_fig": True,
            "spot_size": spotsize,
            "show": False,
        }

        if img_key is None:
            args["img_key"] = None

        if type(obs_name) == list:
            args["color"] = [*obs_name]
        else:
            args["color"] = [obs_name]
        return args

    ### INTERACTION WITH THE COORDINATE_DF

    def coordinates_sf_for_image(self, img_key):
        return self.coordinate_df.coordinates_sf_for_img(img_key)

    def get_image_coordinate_dict(self, img_key, coordinate_id):
        return self.coordinate_df.get_img_coordinate_dict(img_key, coordinate_id)

    def add_coordinates(self, coordinate_id, scalefactor, img_key):
        return self.coordinate_df.add_coordinates(coordinate_id, scalefactor, img_key)

    ### COORDINATES SETTER AND GETTER

    def get_unscaled_coordinates(self, coordinate_id) -> NDArray[np.float_]:
        return self.adata.obsm[f"coords__{coordinate_id}"]

    def get_lims_unscaled_coordinates(self, coordinate_id):
        # returns the limits of the unscaled coordinates
        coords = self.get_unscaled_coordinates(coordinate_id)
        return np.array([np.min(coords[:, 0]), np.max(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 1])])

    def get_coordinates(self, img_key, coordinate_id=None, round = False, truncate = True):
        # find a coordinate_id and a scale factor for img_key
        # return scaled coordinates
        if round and truncate:
            raise ValueError("it is either round or truncate")

        coords = self.get_image_coordinate_dict(img_key, coordinate_id)
        coordinate_id = coords["coordinate_id"]
        coords = (self.get_unscaled_coordinates(coordinate_id) * coords["scalefactor"])
        if round:
            coords = np.round(coords).astype(int)
        elif truncate:
            coords = coords.astype(int) #this does truncation

        return coords

    def set_unscaled_coordinates(self, coordinate_id, coordinates):
        self.adata.obsm[f"coords__{coordinate_id}"] = coordinates

    def add_array_coords_to_obs(self):
        coords = self.get_unscaled_coordinates("array")
        rows = coords[:,0].flatten()
        cols = coords[:,1].flatten()
        self.obs["array_row"] = rows
        self.obs["array_col"] = cols

    def within_img(self, img_key):
        # returns a boolean array of the same length as self.adata.obs.index
        # True if the coordinates are within the image defined by img_key
        img = self.get_img(img_key)
        img_shape = img.shape[:2]
        coords = self.get_coordinates(img_key)
        within = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < img_shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < img_shape[1])
        )
        return within

    ### IMAGE SETTER AND GETTER

    def add_img(
        self, img, img_key, img_path=None, scalefactor=None, in_memory=True, rgb=True
    ):
        # when specified, img_path suggests to save the image !

        if img_path is not None:
            cv2_utils.save_img(img, img_path, rgb) #this potentially breaks if the img is not in int uint8 type...
        if in_memory:
            self.adata.uns["spatial"][self.library_id]["images"][img_key] = img
            self.adata.uns["spatial"][self.library_id]["scalefactors"][
                f"tissue_{img_key}_scalef"
            ] = scalefactor

        self.img_df.add_img(img_key, img_path, scalefactor, in_memory)

    def get_img(self, img_key):
        if self.is_img_in_memory(img_key):
            return self.adata.uns["spatial"][self.library_id]["images"][img_key]
        else:
            img_path = self.get_img_path(img_key)
            return cv2_utils.load_img(
                img_path
            )  # TODO: might need to add rgb here ! maybe to store in img_df ?

    def store_img(self, img_key, img_path):
        img = self.get_img(img_key)
        rgb = is_rgb(img)
        cv2_utils.save_img(img, img_path, rgb)
        self.img_df.add_img_path(img_key, img_path)

    def remove_img_from_memory(self, img_key):
        if self.is_img_in_memory(img_key):
            del self.adata.uns["spatial"][self.library_id]["images"][img_key]
            self.img_df.remove_img_from_memory(img_key)

    ### OTHER IMG UTILITIES

    def is_img_in_memory(self, img_key):
        return self.img_df.is_img_in_memory(img_key)

    def get_img_path(self, img_key):
        return self.img_df.get_img_path(img_key)

    def get_img_scalefactor(self, img_key):
        return self.img_df.get_img_scalefactor(img_key)

    def downscaled_he_img_key(self, mpp = None):
        if mpp is None:
            img_keys = self.coordinate_df.df["img_key"].dropna().tolist()
            img_key_select = [key for key in img_keys if "mpp" in key]
            if len(img_key_select) > 1:
                raise ValueError(
                    "there is more than 1 img with mpp in its key. Therefore downscaled_he_img_key is ambiguous."
                )
            return img_key_select[0]
        else:
            return downscaled_he_img_key(mpp)

    @property
    def source_image_path(self):
        return self.adata.uns["spatial"][self.library_id]["metadata"][
            "source_image_path"
        ]

    ### OTHERS

    @property
    @history_decorator
    def n_counts(self):
        # returns the total counts and updates obs["n_counts"]

        n_counts = np.array(np.sum(self.adata.X, axis=-1)).flatten()
        self.adata.obs["n_counts"] = n_counts
        return n_counts

    @property
    def array_coords(self):
        return self.adata.obsm["coords__array"]

    @property
    def spotsize(self):
        return self.adata.uns["spatial"][self.library_id]["scalefactors"][
            "spot_diameter_fullres"
        ]

    @property
    def obs(self) -> pd.DataFrame: 
        return self.adata.obs

    @property
    def obsm(self): 
        return self.adata.obsm

    @property
    def index(self):
        return self.adata.obs.index

    @property
    def X(self):
        return self.adata.X

    @property
    def shape(self):
        return self.adata.shape

    @property
    def shape_array(self):
        array_coords = self.get_unscaled_coordinates("array")
        max_row, max_col = np.max(array_coords, axis = 0)
        min_row, min_col = np.min(array_coords, axis = 0)
        n_cols = max_col - min_col + 1
        n_rows = max_row - min_row + 1
        return n_rows, n_cols

    def nucl_indices(self, cell_id_label = "cell_id"):
        return self.obs[cell_id_label].dropna().index

    def nucl_mask(self, cell_id_label = "cell_id"):
        return np.logical_not(self.obs[cell_id_label].isna().to_numpy())
