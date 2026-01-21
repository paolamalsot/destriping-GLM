from __future__ import annotations
from typing import TYPE_CHECKING
from src.destriping.simulation.segmentation_mask_generator import SegmentationMask
from src.destriping.simulation.stripe_factors_generator import StripeFactors
from anndata import AnnData
import numpy as np
if TYPE_CHECKING:
    from src.spatialAdata.spatialAdata import spatialAdata
from scipy.sparse import csr_matrix, lil_matrix
from src.spatialAdata.from_numpy import make_sdata
import pandas as pd
from src.utilities.nbinom import convert_n_binom_params
from src.utilities.sparse_utils import indexing_csr
from src.utilities.sparse_utils import convert_to_32_bit
# destriping model under poisson counts

# hyperparameters
# nrows
# ncols
# cell-positions

# generation of cell-positions/segmentation mask. Sparse matrix with at [i,j] the index of the corresponding compartment

# class segmentation mask generator
# init (hyperparameters)
# hyperparameters: occupational density (100%), typical cell-size (bin-count per cell), random_seed
# initialize generator

# generate
# returns a random segmentation mask

# class segmentation mask
# init (matrix)
# save the matrix
# method show()

# avg_count_per_cell


def avg_tot_counts_per_cell(adata: AnnData):
    return np.mean(adata.X.sum(-1))

def expected_counts_per_bin(expected_concentration_gene_profile, bin_df):
    return expected_concentration_gene_profile[bin_df.cell_id]

def poisson_generator_with_csr_mu(mu, generator):
    assert type(mu) is csr_matrix, "mu must be a csr_matrix"
    # generate poisson counts for each non-zero entry in mu
    poisson_counts = generator.poisson(mu.data)
    # create a new csr_matrix with the same shape as mu
    poisson_counts_matrix = csr_matrix((poisson_counts, mu.indices, mu.indptr), shape=mu.shape)
    return poisson_counts_matrix

def negative_binomial_generator_with_csr_mu(mu, generator, dispersion):
    """
    Generates a sparse matrix of negative binomial counts based on the expectation counts matrix.
    """
    assert type(mu) is csr_matrix, "mu must be a csr_matrix"
    # generate negative binomial counts for each non-zero entry in mu
    if dispersion <= 0:
        raise ValueError("dispersion_parameter must be greater than 0")
    n, p = convert_n_binom_params(mu.data, dispersion)
    negative_binomial_counts = generator.negative_binomial(n, p)
    # create a new csr_matrix with the same shape as mu
    negative_binomial_counts_matrix = csr_matrix((negative_binomial_counts, mu.indices, mu.indptr), shape=mu.shape)
    return negative_binomial_counts_matrix

class PoissonCountGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.generator = np.random.default_rng(seed=seed)
    
    def generate(self, expectation_counts_matrix):
        """
        Generates a sparse matrix of Poisson counts based on the expectation counts matrix.
        """
        if not isinstance(expectation_counts_matrix, csr_matrix):
            raise ValueError("expectation_counts_matrix must be a csr_matrix")
        return poisson_generator_with_csr_mu(expectation_counts_matrix, self.generator)

class NegativeBinomialCountGenerator:
    def __init__(self, dispersion, seed):
        self.dispersion = dispersion
        self.seed = seed
        self.generator = np.random.default_rng(seed=seed)

    def generate(self, expectation_counts_matrix):
        if not isinstance(expectation_counts_matrix, csr_matrix):
            raise ValueError("expectation_counts_matrix must be a csr_matrix")
        return negative_binomial_generator_with_csr_mu(expectation_counts_matrix, self.generator, self.dispersion)


def get_count_generator(distribution: str, distribution_params: dict, seed):
    """
    Returns a count generator based on the specified distribution and parameters.
    """
    if distribution == "poisson":
        return PoissonCountGenerator(seed = seed, **distribution_params)
    elif distribution == "negative_binomial":
        return NegativeBinomialCountGenerator(seed = seed, **distribution_params)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")


class SpatialCountGenerator():
    def __init__(self, segmentation_mask: SegmentationMask, 
                 horizontal_stripe_factors: StripeFactors,
                 vertical_stripe_factors: StripeFactors,
                 expected_concentration_gene_profile_: AnnData,
                 avg_tot_counts_per_bin: float | None = None, 
                 sigma_tot_counts_per_bin_across_cells: float | None = None,
                 random_seed: int | None = None,
                 distribution: str = "poisson",
                 distribution_params: dict | None = None):

        """
        - avg_tot_counts_per_bin (float): on average across cells, the expected total counts per bin. NB: the actual value will differ, since the values are poisson generated...
        - sigma_tot_counts_per_bin_across_cells (float): std of the expected tot counts per bin across cells
        """
        self.distribution = distribution
        self.distribution_params = distribution_params if distribution_params is not None else {}

        self.segmentation_mask = segmentation_mask
        self.expected_concentration_gene_profile_ = expected_concentration_gene_profile_

        self.avg_tot_counts_per_bin = avg_tot_counts_per_bin
        self.sigma_tot_counts_per_bin_across_cells = sigma_tot_counts_per_bin_across_cells

        self.horizontal_stripe_factors = horizontal_stripe_factors
        self.vertical_stripe_factors = vertical_stripe_factors

        assert segmentation_mask.shape == (len(self.horizontal_stripe_factors), len(self.vertical_stripe_factors))
        if segmentation_mask.n_cells != self.expected_concentration_gene_profile_.shape[0]:
            raise ValueError("Different number of cells in expected_concentration_gene_profile_ and in segmentation_mask")

        self.random_seed = random_seed
        self.generator = np.random.default_rng(seed=random_seed)

    @property
    def n_cells(self):
        return self.segmentation_mask.n_cells

    def generate(self) -> tuple[spatialAdata, AnnData, spatialAdata, spatialAdata]:
        """
        Returns:
        - generated spatial data n_bins x n_genes
        - AnnData cell_id x n_genes with the unstriped expression profile
        """

        # expected_concentration_gene_profile_ is now passed in, not computed here
        expected_concentration_gene_profile_ = self.expected_concentration_gene_profile_ #n_cells x n_genes
        expected_concentration_gene_profile_.X = convert_to_32_bit(
            expected_concentration_gene_profile_.X
        )

        bin_df = self.segmentation_mask.bin_df.copy()
        # convert bin_df cell_id (num) to the cell_gene_expression_profile cell_id
        nulei_indices =  self.segmentation_mask.nuclei_bin_df.index
        # alignment between expected_concentration_gene_profile_ and segmentation mask
        unique_categories, int_cell_id = np.unique(self.segmentation_mask.nuclei_bin_df["cell_id"], return_inverse=True)
        bin_df.loc[nulei_indices, "reference_index"] = self.expected_concentration_gene_profile_.obs['reference_index'].iloc[int_cell_id].values #not so sure it will work

        # filling nuclei
        n_obs = len(bin_df)
        n_genes = self.expected_concentration_gene_profile_.shape[-1]
        expectation_counts_matrix_unstriped = lil_matrix((n_obs, n_genes), dtype = np.float32)
        # https://stackoverflow.com/a/20344429
        expectation_counts_matrix_unstriped[nulei_indices] = indexing_csr(expected_concentration_gene_profile_.X,int_cell_id, 100000)
        expectation_counts_matrix_unstriped = expectation_counts_matrix_unstriped.tocsr()

        expectation_counts_matrix = expectation_counts_matrix_unstriped\
            .multiply(self.horizontal_stripe_factors[bin_df.row_idx].reshape(-1,1))\
            .multiply(self.vertical_stripe_factors[bin_df.col_idx].reshape(-1,1))

        # expectation_counts_matrix = expectation_counts_matrix.todense() #TODO: is this necessary ?
        bin_df["n_counts_expected_with_stripes"] = expectation_counts_matrix.sum(-1)
        bin_df["n_counts_expected_without_stripes"] = expectation_counts_matrix_unstriped.sum(-1)
        bin_df["stripe_factor"] = (self.horizontal_stripe_factors[bin_df.row_idx].toarray() * self.vertical_stripe_factors[bin_df.col_idx].toarray())

        # generated_counts_matrix = self.generator.poisson(expectation_counts_matrix)
        expectation_counts_matrix = csr_matrix(expectation_counts_matrix, dtype = np.float32)
        count_generator = get_count_generator(self.distribution, self.distribution_params, self.random_seed)
        generated_counts_matrix = convert_to_32_bit(count_generator.generate(expectation_counts_matrix))
        array_coordinates = bin_df.loc[:, ["row_idx", "col_idx"]].values # top down, left to right
        bin_df.index = bin_df.index.astype("str")  
        var = self.expected_concentration_gene_profile_.var
        spatialdata = make_sdata(array_coordinates, generated_counts_matrix, obs = bin_df, var = var)
        expected_sdata = make_sdata(
            array_coordinates,
            expectation_counts_matrix,
            obs=bin_df,
            var=var,
        )
        expected_sdata_wo_stripes = make_sdata(
            array_coordinates,
            expectation_counts_matrix_unstriped,
            obs=bin_df,
            var=var,
        )

        obs = pd.DataFrame(data = {"cell_id": unique_categories, "reference_index": self.expected_concentration_gene_profile_.obs['reference_index'].values})
        obs.index = obs.index.astype('str')
        cell_gene_expression_profile = AnnData(
            X=expected_concentration_gene_profile_.X,
            obs=obs,
            var=var,
        )
        cell_gene_expression_profile.obs.index = cell_gene_expression_profile.obs["cell_id"].values

        return spatialdata, cell_gene_expression_profile, expected_sdata, expected_sdata_wo_stripes
