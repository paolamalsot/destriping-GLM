from src.destriping.simulation.simulator import generate_spatial_data
from hydra.utils import instantiate
import scipy.sparse as sp
import os
import pandas as pd
from src.destriping.sol import Sol

def parse_config(original_root, cfg):

    main_output_dir = os.getcwd()
    simulation_params = cfg.simulation_params
    if simulation_params["sc_ref_path"] is not None:
        simulation_params["sc_ref_path"] = os.path.join(original_root, simulation_params["sc_ref_path"])
    if simulation_params["segmentation_mask_path"] is not None:
        simulation_params["segmentation_mask_path"] = os.path.join(original_root, simulation_params["segmentation_mask_path"])
    if "expected_concentration_gene_profile_path" in simulation_params.keys() and simulation_params["expected_concentration_gene_profile_path"] is not None:
        simulation_params["expected_concentration_gene_profile_path"] = os.path.join(original_root, simulation_params["expected_concentration_gene_profile_path"])
    simulation_params = instantiate(simulation_params, _convert_="all")
    spatial_data_generator = generate_spatial_data(**simulation_params)
    spatial_data, cell_gene_expression, expected_sdata, expected_sdata_wo_stripes = spatial_data_generator.generate()
    
    for d_ in [spatial_data, expected_sdata_wo_stripes]:
        d_.adata.X = sp.csr_matrix(d_.adata.X)

    spatial_data_path = os.path.join(main_output_dir, "spatial_data")
    spatial_data.save(spatial_data_path)

    gt_sol = Sol(pd.Series(spatial_data_generator.horizontal_stripe_factors),
                        pd.Series(spatial_data_generator.vertical_stripe_factors),
                        expected_sdata_wo_stripes)
    output_dir = os.path.join(main_output_dir, "gt_sol")
    os.makedirs(output_dir, exist_ok= True)
    gt_sol.save(output_dir)

    expected_sdata_wo_stripes_path = os.path.join(main_output_dir, "expected_spatial_data_wo_stripes")
    expected_sdata_wo_stripes.save(expected_sdata_wo_stripes_path)

    cell_gene_expression_path = os.path.join(main_output_dir, "cell_gene_expression.h5ad")
    cell_gene_expression.write(filename=cell_gene_expression_path)