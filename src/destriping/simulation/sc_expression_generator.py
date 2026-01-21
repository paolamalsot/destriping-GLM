import numpy as np
import anndata
import scanpy as sc


class Sc_expression_generator:
    def __init__(self, n_cells, n_genes, reference: anndata.AnnData, random_seed=None):
        """
        Initialize the Sc_expression_generator.

        Parameters:
        - n_cells (int): Number of cells to sample.
        - n_genes (int): Number of highly variable genes to include in the generated data.
        - reference (anndata.AnnData): The reference single-cell AnnData object with expression data.
        - random_seed (int or None): Seed for random number generator, default is None.
        """
        self.n_cells = n_cells
        self.n_genes = n_genes
        self.reference = reference  # AnnData object n_obs x n_vars sparse matrix

        if n_cells > self.reference.shape[0]:
            raise ValueError(
                f"n_cells ({n_cells}) cannot be greater than the number of cells in the reference ({self.reference.shape[0]})"
            )
        if n_genes > self.reference.shape[1]:
            raise ValueError(
                f"n_genes ({n_genes}) cannot be greater than the number of genes in the reference ({self.reference.shape[1]})"
            )

        self.generator = np.random.default_rng(seed=random_seed)

    def generate(self) -> anndata.AnnData:
        """
        Generate a new expression matrix by randomly selecting cells and picking the most variable genes.
        Note that sampled data will have str(int) indices, and the "reference_index" column in obs will contain the corresponding sampled cells indices.
        """
        # Randomly sample indices of cells without repetition
        cell_indices = self.generator.choice(
            self.reference.shape[0], size=self.n_cells, replace=False
        )

        # Identify highly variable genes
        sc.pp.highly_variable_genes(
            self.reference, n_top_genes=self.n_genes, inplace=True, flavor="seurat_v3"
        )

        # Get the indices of the most variable genes
        variable_gene_indices = np.where(self.reference.var["highly_variable"])[0]

        # Extract the submatrix from the reference data using the sampled cell and variable gene indices
        sampled_matrix = self.reference[cell_indices, variable_gene_indices].copy()

        # obs
        obs = self.reference.obs.iloc[cell_indices].copy()
        obs.reset_index(names="reference_index", inplace=True)
        obs.index = obs.index.astype(str)  # to avoid warning casting to str

        # var
        var = self.reference.var.copy()
        var = var.iloc[variable_gene_indices][["gene_ids"]]

        # Return a new AnnData object containing the sampled cells and highly variable genes
        sampled_data = anndata.AnnData(X=sampled_matrix.X, obs=obs, var=var)

        return sampled_data
