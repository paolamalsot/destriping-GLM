from functools import partial
from src.spatialAdata.spatialAdata import spatialAdata
import logging
from pathlib import Path
import yaml


def make_path_rel_root_lambda(original_root):
    def path_rel_root(x):  # .resolve(strict=False) prevents following symlinks
        return Path(x).absolute().relative_to(Path(original_root).absolute()).as_posix()
        # return os.path.normpath(os.path.relpath(os.path.abspath(x), original_root))

    return path_rel_root


def generating_data_distribution(path_data):
    dir = Path(path_data).parent
    hydra_config_path = dir / ".hydra/config.yaml"
    simulation_params = yaml.safe_load(hydra_config_path.read_text())[
        "simulation_params"
    ]
    if (
        "distribution" in simulation_params.keys()
        and "distribution_params" in simulation_params.keys()
    ):
        return (
            simulation_params["distribution"],
            simulation_params["distribution_params"],
        )
    else:  # default
        return "poisson", {}


def convert_gen_params_to_qm_params(distribution, distribution_params):
    if distribution == "negative_binomial":
        r = distribution_params["dispersion"]
        new_distribution_params = {"r": r}
        distribution = "nbinom"
        return distribution, new_distribution_params
    else:
        return distribution, distribution_params


def args_combi_nucl_cyto(sol):
    nucl_args = {
        "row_factors": sol.h,
        "col_factors": sol.w,
        "gene_expression_per_bin": sol.c,
    }
    cyto_args = {"row_factors": sol.h, "col_factors": sol.w}
    return {"args_nucl": nucl_args, "args_cyto": cyto_args}


def get_args_for_method(method, sol, dist="poisson", dist_params=None):
    if method.__name__ == spatialAdata.destripe_dividing_factors.__name__:
        return {"row_factors": sol.h, "col_factors": sol.w}
    elif method.__name__ in [
        spatialAdata.destripe_dividing_factors_qm_tot_counts.__name__
    ]:
        return {
            "row_factors": sol.h,
            "col_factors": sol.w,
            "gene_expression_per_bin": sol.c,
            "dist": dist,
            "dist_params": dist_params,
        }
    else:
        raise ValueError("Wrong method name...")


def get_all_destripe_calls(
    sol, model, data, cell_id_label, dist="poisson", dist_params=None
):
    # returns a list of methods we can simply call to get the destriped data !
    # destriping
    nucl_indices = data.nucl_indices(cell_id_label=cell_id_label)

    # does the method has c for every bin ?
    c_all = len(sol.c.adata) == len(data.adata)
    dict_calls = {}

    methods_list = [
        spatialAdata.destripe_dividing_factors,
        spatialAdata.destripe_dividing_factors_qm_tot_counts,
    ]

    logging.info(methods_list)
    logging.info(f"c_all: {c_all}")

    for method in methods_list:
        if c_all:
            name_method = f"{method.__name__}"
            args = get_args_for_method(method, sol, dist=dist, dist_params=dist_params)
            dict_calls[name_method] = partial(getattr(data.copy(), name_method), **args)

        else:
            cyto_method = spatialAdata.destripe_dividing_factors
            nucl_method = method
            name_method = f"cyto_{cyto_method.__name__}_nucl_{method.__name__}"  # how to get just the last part ?
            args_nucl = get_args_for_method(
                nucl_method, sol, dist=dist, dist_params=dist_params
            )
            args_cyto = get_args_for_method(
                cyto_method, sol, dist=dist, dist_params=dist_params
            )

            def apply_after_copy(data_, method_name, args_dict_):
                u = data_.copy()
                return getattr(u, method_name)(**args_dict_)

            args_dict = {
                "method_cyto": cyto_method,
                "method_nucl": nucl_method,
                "nucl_indices": nucl_indices,
                "args_nucl": args_nucl,
                "args_cyto": args_cyto,
            }
            dict_calls[name_method] = partial(
                apply_after_copy,
                data_=data,
                method_name="destripe_combi_nucl_cyto",
                args_dict_=args_dict,
            )

    return dict_calls
