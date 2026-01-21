import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from scipy.ndimage import label
from src.destriping.simulation.random_cell_mask import random_shapes
import warnings
import logging
from numpy.typing import NDArray
import src.spatialAdata.labels_convention as labels_convention

class SegmentationMask:
    def __init__(self, matrix : NDArray[object]):
        """
        - matrix (2D str np.array) with str representing cell_id and nan standing for unassigned bin.
        """
        self.matrix = matrix
        #self.verify_matrix()

    # def verify_matrix(self):
    #     max_val = np.max(self.matrix)
    #     min_val = np.min(self.matrix)
    #     unique_values = np.unique(self.matrix)
    #     assert np.all(unique_values == np.arange(min_val, max_val+1))

    def show(self):
        """Display the segmentation mask."""
        plt.figure(figsize=(6, 6))
        to_plot, cmap = labels_convention.convert_str_to_rgb(self.matrix)
        plt.imshow(to_plot, interpolation='none')
        plt.title('Segmentation Mask')
        plt.show()

    @property
    def all_lanes_occupied(self) -> bool:
        nucleic_bin_df = self.nuclei_bin_df
        all_bin_df = self.bin_df
        all_rows_occupied = len(nucleic_bin_df["row_idx"].value_counts()) == len(all_bin_df["row_idx"].value_counts())
        all_cols_occupied = len(nucleic_bin_df["col_idx"].value_counts()) == len(all_bin_df["col_idx"].value_counts())
        all_occupied = all_rows_occupied and all_cols_occupied
        return all_occupied

    @property
    def n_cells(self) -> int:
        return len(np.unique(self.matrix[~(pd.isna(self.matrix))]))
    
    @property
    def shape(self) -> tuple[int,int]:
        return self.matrix.shape
    
    @property
    def bin_df(self) -> pd.DataFrame:
        indices, values = zip(*np.ndenumerate(self.matrix))
        row_idx, col_idx = zip(*indices)
        row_idx = np.array(row_idx)
        col_idx = np.array(col_idx)
        values = np.array(values, dtype = object)
        #oi = (values != -1) #remove the bins that do not belong to any cell
        #row_idx = row_idx[oi]
        #col_idx = col_idx[oi]
        #values = values[oi]
        
        #to avoid mistakes, convert integer cell_id to random strings...
        
        return pd.DataFrame(data = {"row_idx": row_idx,
                                    "col_idx": col_idx,
                                    "cell_id": pd.Categorical(values)})
    
    @property
    def nuclei_bin_df(self) -> pd.DataFrame:
        bin_df = self.bin_df
        return bin_df[bin_df["cell_id"].notna()]
    

def get_range_diameter_from_range_size(min_size, max_size):
    max_radius = np.sqrt(max_size/np.pi) * 2
    min_radius = np.sqrt(min_size/np.pi) * 2
    return min_radius, max_radius


class SegmentationMaskGenerator:
    def __init__(self, shape: tuple[int, int], occupational_density: float, min_cell_size: int, max_cell_size:int, n_trials_all_lanes_occupied: int = 3, n_trials_shape: int = 50, random_seed: int = None):
        """
        Initialize the SegmentationMaskGenerator.
        
        Parameters:
        - shape: Shape of the segmentation mask (nrows, ncols)
        - occupational_density (float): fraction of bins occupied by a cell
        - min_cell_size (int): minimum number of bins of a cell
        - max_cell_size (int): maximum number of bins of a cell
        - n_trials_all_lanes_occupied (int): number of iterations to try to occupy all lanes (i.e not getting empty lanes)
        - n_trials_shape (int): number of iterations to fit a shape
        """
        self.shape = shape
        self.occupational_density = occupational_density
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size 
        self.random_seed = random_seed
        self.generator = np.random.default_rng(seed=random_seed)
        self.n_trials_shape = n_trials_shape
        self.n_trials_all_lanes_occupied = n_trials_all_lanes_occupied 

    def generate(self) -> SegmentationMask:
        nrows, ncols = self.shape
        total_bins = nrows * ncols
        typical_cell_size = (self.max_cell_size + self.min_cell_size)/2
        n_shapes = np.floor(self.occupational_density * total_bins/typical_cell_size)

        logging.debug(f"{n_shapes=}")

        min_diam, max_diam = get_range_diameter_from_range_size(self.min_cell_size, self.max_cell_size)

        for trial in range(self.n_trials_all_lanes_occupied):
            with warnings.catch_warnings(record=True):
                img, _ = random_shapes(self.shape, 
                                n_shapes,
                                min_shapes =  n_shapes,
                                min_size = min_diam, 
                                max_size = max_diam, 
                                shape = "ellipse", 
                                rng = self.generator, 
                                allow_overlap=False,
                                num_trials = self.n_trials_shape)
            
            img = labels_convention.int_to_word(img)
            mask = SegmentationMask(img)
            
            all_lanes_occupied = mask.all_lanes_occupied
            if all_lanes_occupied:
                break

        if not(all_lanes_occupied):
            warnings.warn(f"After {self.n_trials_all_lanes_occupied} trials, did not manage to occupy all lanes.")
        
        return mask

# Example usage
if __name__ == '__main__':
    generator = SegmentationMaskGenerator(shape=(100, 100), 
                                          occupational_density=0.8, 
                                          min_cell_size=5, 
                                          max_cell_size=10, 
                                          random_seed=42)
    mask = generator.generate()
    mask.show()