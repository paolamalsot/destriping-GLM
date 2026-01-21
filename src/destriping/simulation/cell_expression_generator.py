import numpy as np

def compute_expected_concentration_gene_profile(cell_gene_expression_profile, counts_per_cell):
    n_cells = cell_gene_expression_profile.shape[0]
    # normalization concentration
    cell_gene_expression_concentration = (cell_gene_expression_profile / np.sum(cell_gene_expression_profile, axis = -1)).multiply(counts_per_cell[np.arange(n_cells)].reshape(-1,1))
    return cell_gene_expression_concentration.tocsr() #concentration in counts per bin

class ExpectedConcentrationGenerator:
    def __init__(self, cell_gene_expression_profile, avg_tot_counts_per_bin, sigma_tot_counts_per_bin_across_cells, random_seed: int | None = None):
        self.cell_gene_expression_profile = cell_gene_expression_profile
        self.avg_tot_counts_per_bin = avg_tot_counts_per_bin
        self.sigma_tot_counts_per_bin_across_cells = sigma_tot_counts_per_bin_across_cells
        self.generator = np.random.default_rng(seed=random_seed)

    @property 
    def n_cells(self):
        return self.cell_gene_expression_profile.shape[0]

    def generate(self):
        #generate_tot_counts_per_cell
        counts_per_cell = np.clip(self.generator.normal(loc = self.avg_tot_counts_per_bin, 
                                                scale = self.sigma_tot_counts_per_bin_across_cells, 
                                                size = self.n_cells), a_min = 0, a_max = None)
        expected_concentration = compute_expected_concentration_gene_profile(self.cell_gene_expression_profile, counts_per_cell)
        return expected_concentration
                