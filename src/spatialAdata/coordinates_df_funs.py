import numpy as np
import pandas as pd
from functools import reduce


def selector_within_limits(coordinates, limits):
    """
    coordinates is a Nx2 np array
    limits is a length 4 array that specifies min(coordinates[:,0]), max(coordinates[:,0]), min(coordinates[:,1]), max(coordinates[:,1])
    returns a boolean selection array indicating which of the N coordinates are within limits
    """
    min_0, max_0, min_1, max_1 = limits

    # List of conditions
    conditions = [
        coordinates[:, 0] >= min_0,
        coordinates[:, 0] <= max_0,
        coordinates[:, 1] >= min_1,
        coordinates[:, 1] <= max_1,
    ]

    # Applying np.logical_and across all conditions using reduce
    selector = reduce(np.logical_and, conditions)

    return selector


def get_unscaled_coordinates(adata, coordinate_id):
    return adata.obsm["coords__" + coordinate_id]


def convert_unscaled_limits_basis(
    adata, limits_basis_A, coordinate_id_basis_A, coordinate_id_basis_B
):
    # if the coordinates are tilted wrt each other, we return the smallest possible region than encloses the one specified by limits_basis_A, such that is is a rectangular region parallel to the axis of basis_B

    coords_basis_A = get_unscaled_coordinates(adata, coordinate_id_basis_A)
    selector_array = selector_within_limits(coords_basis_A, limits_basis_A)
    coords_basis_B = get_unscaled_coordinates(adata, coordinate_id_basis_B)[
        selector_array, :
    ]
    min_0 = np.min(coords_basis_B[:, 0])
    max_0 = np.max(coords_basis_B[:, 0])
    min_1 = np.min(coords_basis_B[:, 1])
    max_1 = np.max(coords_basis_B[:, 1])
    new_limits = min_0, max_0, min_1, max_1
    return new_limits


def scale_limits(limits, scale_factor):
    return (np.array(limits) * scale_factor).tolist()


def get_limits_pixel_coordinates(
    img_key, coordinate_id, unscaled_limits, coordinate_df, sdata
):
    img_coordinate_id, scale_factor = coordinate_df.coordinates_sf_for_img(img_key)
    limits_coords = convert_unscaled_limits_basis(
        sdata.adata, unscaled_limits, coordinate_id, img_coordinate_id
    )
    limits_pixel_coords = np.array(scale_limits(limits_coords, scale_factor))
    limits_pixel_coords[[0, 2]] = np.floor(limits_pixel_coords[[0, 2]])
    limits_pixel_coords[[1, 3]] = np.ceil(limits_pixel_coords[[1, 3]])
    return limits_pixel_coords.astype(int).tolist()
