from src.destriping.simulation.sc_expression_generator import Sc_expression_generator
from src.destriping.simulation.segmentation_mask_generator import SegmentationMaskGenerator, SegmentationMask
from src.destriping.simulation.poisson_count_generator import SpatialCountGenerator
from src.destriping.simulation.stripe_factors_generator import WeibullStripeFactorsGeneratorConstrained, plot_stripes, statistics_stripes
from src.destriping.simulation.cell_expression_generator import ExpectedConcentrationGenerator
import numpy as np
from os import PathLike
import scanpy as sc
import anndata

generator_map = {
    "weibull_constrained": WeibullStripeFactorsGeneratorConstrained
}

def generator_cls_from_str(generator_name: str) -> type[WeibullStripeFactorsGeneratorConstrained]:

    if generator_name not in generator_map:
        raise ValueError(f"Unknown stripe_factors_generator: {generator_name}")
    generator_cls = generator_map[generator_name]
    return generator_cls


def generate_spatial_data(
    random_seed: int,
    n_rows: int | None,
    n_cols: int | None,
    occupational_density: float | None,
    min_cell_size: int | None,
    max_cell_size: int | None,
    n_genes: int,
    sc_ref_path: str,
    avg_tot_counts_per_bin: float,
    sigma_tot_counts_per_bin_across_cells: float,
    segmentation_mask_path: str | PathLike | None = None,
    stripe_factors_generator: str | tuple = "gaussian_constrained",
    stripes_generator_params: dict | tuple = None,
    expected_concentration_gene_profile_path: str | None = None,
    distribution: str = "poisson",
    distribution_params: dict = None
) -> SpatialCountGenerator:
    # intuitively, we would think that the parameter n_genes is not important. However because the reference concentration is taken from single-cell-data, with only two genes, there is super high chance to get 0 total counts in the cell.

    #generate the seeds
    if segmentation_mask_path is not None:
        mask = SegmentationMask(np.load(segmentation_mask_path, allow_pickle=True))
        n_rows, n_cols = mask.shape
    else:
        generator = SegmentationMaskGenerator(shape=(n_rows, n_cols), 
                                            occupational_density=occupational_density, 
                                            min_cell_size=min_cell_size, 
                                            max_cell_size=max_cell_size, 
                                            random_seed=random_seed)

        mask = generator.generate()

    n_cells = mask.n_cells

    if expected_concentration_gene_profile_path is not None:
        expected_concentration_gene_profile_ = anndata.read_h5ad(expected_concentration_gene_profile_path)
    else:
        ref = sc.read(sc_ref_path)
        expression_generator = Sc_expression_generator(n_cells=n_cells, n_genes=n_genes, reference=ref, random_seed = random_seed)
        expr_data = expression_generator.generate()
        expected_concentration_gene_profile_generator = ExpectedConcentrationGenerator(expr_data.X, avg_tot_counts_per_bin, sigma_tot_counts_per_bin_across_cells, random_seed)
        expected_concentration_gene_profile_X = expected_concentration_gene_profile_generator.generate()
        expected_concentration_gene_profile_ = expr_data.copy()
        expected_concentration_gene_profile_.X = expected_concentration_gene_profile_X

    if stripes_generator_params is None:
        stripes_generator_params = {}

    # Handle tuple or single value for stripe_factors_generator
    if isinstance(stripe_factors_generator, (tuple, list)):
        horizontal_generator_name = stripe_factors_generator[0]
        vertical_generator_name = stripe_factors_generator[1]
    else:
        horizontal_generator_name = vertical_generator_name = stripe_factors_generator

    # Handle tuple or single value for stripes_generator_params
    if isinstance(stripes_generator_params, (tuple, list)):
        horizontal_stripes_params = stripes_generator_params[0] if stripes_generator_params[0] is not None else {}
        vertical_stripes_params = stripes_generator_params[1] if stripes_generator_params[1] is not None else {}
    else:
        horizontal_stripes_params = vertical_stripes_params = stripes_generator_params if stripes_generator_params is not None else {}

    horizontal_generator_cls = generator_cls_from_str(horizontal_generator_name)
    vertical_generator_cls = generator_cls_from_str(vertical_generator_name)

    horizontal_stripes = horizontal_generator_cls(n_rows, **horizontal_stripes_params, random_seed=random_seed).generate()
    vertical_stripes = vertical_generator_cls(n_cols, **vertical_stripes_params, random_seed=random_seed*2).generate()

    spatial_data_generator = SpatialCountGenerator(
        segmentation_mask = mask,
        horizontal_stripe_factors = horizontal_stripes,
        vertical_stripe_factors = vertical_stripes,
        expected_concentration_gene_profile_ = expected_concentration_gene_profile_,
        avg_tot_counts_per_bin = avg_tot_counts_per_bin,
        sigma_tot_counts_per_bin_across_cells = sigma_tot_counts_per_bin_across_cells,
        random_seed = random_seed,
        distribution= distribution,
        distribution_params= distribution_params
    )

    return spatial_data_generator