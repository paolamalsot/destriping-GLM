import scipy.stats as stats
import numpy as np
import logging

def check_correlation_array_coord_with_img_coord(coords_array, coords_img):
    coord_0_corr = stats.pearsonr(coords_img[:,0], coords_array[:,0])
    coord_1_corr = stats.pearsonr(coords_img[:,1], coords_array[:,1])
    logging.info(f"Correlation between image coordinates and array coordinates for coord 0 (corr, pval): {coord_0_corr}")
    logging.info(f"Correlation between image coordinates and array coordinates for coord 1 (corr, pval): {coord_1_corr}")
    return coord_0_corr, coord_1_corr

def flip_array_coords(coords, coord_index):
    max_array_col = np.max(coords[:, coord_index])
    coords[:, coord_index] = max_array_col - coords[:, coord_index]
    return coords

def reorient_coords(array_coords, img_coords):
    """
    Reorient the array coordinates to match the image coordinates orientation.
    (Generally the img coords are top down left to right)
    """
    # Check correlation before flipping
    coord_0_corr, coord_1_corr = check_correlation_array_coord_with_img_coord(array_coords, img_coords)
    
    if coord_0_corr[0] < 0:  # If correlation is negative, flip the first coordinate
        array_coords = flip_array_coords(array_coords, 0)
    
    if coord_1_corr[0] < 0:  # If correlation is negative, flip the second coordinate
        array_coords = flip_array_coords(array_coords, 1)
    
    return array_coords