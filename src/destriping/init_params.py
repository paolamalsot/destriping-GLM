import numpy as np
import pandas as pd
from src.spatialAdata.spatialAdata import spatialAdata


def quantile_init(
    array_row,
    n_counts,
    quant_init,
    rows_idx_opt=None,
    epsilon=0.0,
    nucl_selector=None,
    nucl_only=False,
):
    """
    Initialize the parameters for the destriping algorithm using quantiles.

    Parameters:
    - array_row (np.ndarray): Array indicating the row indices of the data.
    - n_counts (np.ndarray): Array of counts for each data point.
    - quant_init (float): Quantile value to initialize the factors.
    - rows_idx_opt (np.ndarray, optional): Indices of rows to normalize. Default is None.
    - epsilon (float, optional): Minimum value for the factors. Default is 0.
    - nucl_selector (np.ndarray, optional): Boolean array indicating nuclear observations. Default is None.
    - nucl_only (bool, optional): If True, compute quantiles only on nuclear data. Default is False.

    Returns:
    - h (pd.Series): Row factors as a pandas Series, with other factors set to one.
    """
    # Create a DataFrame for processing
    data = pd.DataFrame({"array_row": array_row, "n_counts": n_counts})

    # Filter data if nucl_only is True
    if nucl_only:
        if nucl_selector is None:
            raise ValueError("nucl_selector must be provided when nucl_only is True.")
        filtered_data = data[nucl_selector]
    else:
        filtered_data = data

    # Compute row factors
    h = filtered_data.groupby("array_row")["n_counts"].quantile(q=quant_init)
    if rows_idx_opt is not None:
        # if some rows_idx are missing, fill them with ones (this can happen for example if we do quantiles_nucl init and rows_idx_opt contain rows wo any nucl)
        h_opt = h.get(rows_idx_opt, default=1.0)
        # h_opt = h[rows_idx_opt].copy()
    else:
        h_opt = h.copy()
    h_opt /= np.mean(h_opt)
    h[:] = 1  # Set all factors to 1
    h.update(h_opt)  # Update only the selected indices
    h = h.reindex(data["array_row"].unique()).fillna(1)  # Reindex on original data
    h = h.clip(lower=epsilon)

    return h


def sum_init(
    array_row,
    n_counts,
    fun: callable,
    rows_idx_opt=None,
    epsilon=0.0,
    nucl_selector=None,
    nucl_only=False,
):
    fun = lambda x: np.sum(x)
    return params_init_with_fun(
        array_row, n_counts, fun, rows_idx_opt, epsilon, nucl_selector, nucl_only
    )


def custom_fun_init(
    array_row,
    n_counts,
    fun: callable,
    rows_idx_opt=None,
    epsilon=0.0,
    nucl_selector=None,
    nucl_only=False,
):
    """
    Initialize the parameters for the destriping algorithm using a lane-specific function called fun.

    Parameters:
    - array_row (np.ndarray): Array indicating the row indices of the data.
    - n_counts (np.ndarray): Array of counts for each data point.
    - fun (callable): name of the function to apply. takes as argument a series/numpy array
    - rows_idx_opt (np.ndarray, optional): Indices of rows to normalize. Default is None.
    - epsilon (float, optional): Minimum value for the factors. Default is 0.
    - nucl_selector (np.ndarray, optional): Boolean array indicating nuclear observations. Default is None.
    - nucl_only (bool, optional): If True, compute quantiles only on nuclear data. Default is False.

    Returns:
    - h (pd.Series): Row factors as a pandas Series, with other factors set to one.
    """

    # Create a DataFrame for processing
    data = pd.DataFrame({"array_row": array_row, "n_counts": n_counts})

    # Filter data if nucl_only is True
    if nucl_only:
        if nucl_selector is None:
            raise ValueError("nucl_selector must be provided when nucl_only is True.")
        filtered_data = data[nucl_selector]
    else:
        filtered_data = data

    # Compute row factors
    h = filtered_data.groupby("array_row")["n_counts"].apply(fun)
    if rows_idx_opt is not None:
        # if some rows_idx are missing, fill them with ones (this can happen for example if we do quantiles_nucl init and rows_idx_opt contain rows wo any nucl)
        h_opt = h.get(rows_idx_opt, default=1.0)
        # h_opt = h[rows_idx_opt].copy()
    else:
        h_opt = h.copy()
    h_opt /= np.mean(h_opt)
    h[:] = 1  # Set all factors to 1
    h.update(h_opt)  # Update only the selected indices
    h = h.reindex(data["array_row"].unique()).fillna(1)  # Reindex on original data
    h = h.clip(lower=epsilon)

    return h


def ones_init(array_row):
    return pd.Series(1, index=np.unique(array_row))


# def quantile_init_nucl(data: spatialAdata, quant_init, rows_idx_opt = None, cols_idx_opt = None, return_numpy = True, cell_id_label = "cell_id"):
#     # add row and col to adata
#     data.add_array_coords_to_obs()
#     nucl_data = data[data.obs.loc[~data.obs[cell_id_label].isna()].index]

#     h = nucl_data.obs.groupby("array_row")["n_counts"].quantile(q = quant_init)
#     if not(rows_idx_opt is None):
#         h = h[rows_idx_opt]
#     h /= np.mean(h)
#     # fill with ones
#     h = h.reindex(data.obs["array_row"].unique()).fillna(1)

#     w = nucl_data.obs.groupby("array_col")["n_counts"].quantile(q = quant_init)
#     if not(cols_idx_opt is None):
#         w = w[cols_idx_opt]
#     w /= np.mean(w)
#     w = w.reindex(data.obs["array_col"].unique()).fillna(1)

#     if return_numpy:
#         w = w.values
#         h = h.values

#     return h, w


from typing import Callable, Tuple, Any
from functools import wraps
import pandas as pd

SeriesPair = Tuple[pd.Series, pd.Series]


def two_stage_nonzero_factors_filter(
    func: Callable[..., SeriesPair]
) -> Callable[..., SeriesPair]:
    """
    Decorator:
      1) call `func(**kwargs)` to get (row_factors, col_factors)
      2) compute non-zero indices
      3) call `func(data=data, **kwargs, rows_idx_opt=..., cols_idx_opt=...)`
    """

    @wraps(func)
    def wrapper(*, data: Any = None, **kwargs: Any) -> SeriesPair:
        # First call (no data, just kwargs)
        row_factors, col_factors = func(data=data, **kwargs)

        # Compute non-zero indices
        rows_idx_opt = row_factors[row_factors != 0].index
        cols_idx_opt = col_factors[col_factors != 0].index

        # Second call with augmented kwargs (data included here)
        final_kwargs = {
            **kwargs,
            "rows_idx_opt": rows_idx_opt,
            "cols_idx_opt": cols_idx_opt,
            "data": data,
        }
        return func(**final_kwargs)

    return wrapper


def quantile_init_from_sdata(
    data: spatialAdata,
    quant_init,
    rows_idx_opt=None,
    cols_idx_opt=None,
    epsilon=0,
    nucl_only=False,
    cell_id_label="cell_id",
):
    """
    Adapter function to format input from a spatialAdata object for quantile_init.

    Parameters:
    - data (spatialAdata): The spatialAdata object containing the data.
    - quant_init (float): Quantile value to initialize the factors.
    - rows_idx_opt (np.ndarray, optional): Indices of rows to normalize. Default is None.
    - cols_idx_opt (np.ndarray, optional): Indices of columns to normalize. Default is None.
    - epsilon (float, optional): Minimum value for the factors. Default is 0.
    - nucl_only (bool, optional): If True, compute quantiles only on nuclear data. Default is False.
    - cell_id_label (str, optional): Label for cell IDs in the data. Default is "cell_id".

    Returns:
    - h (pd.Series): Row factors as a pandas Series, with other factors set to one.
    - w (pd.Series): Column factors as a pandas Series, with other factors set to one.
    """
    # Add row and column coordinates to the data
    data.n_counts
    data.add_array_coords_to_obs()

    # Extract necessary fields
    array_row = data.obs["array_row"].values
    array_cols = data.obs["array_col"].values
    n_counts = data.obs["n_counts"].values

    # Create a nuclear selector if needed
    nucl_selector = None
    if nucl_only:
        nucl_selector = ~data.obs[cell_id_label].isna().values

    # Call quantile_init with formatted inputs
    row_factors = quantile_init(
        array_row=array_row,
        n_counts=n_counts,
        quant_init=quant_init,
        rows_idx_opt=rows_idx_opt,
        epsilon=epsilon,
        nucl_selector=nucl_selector,
        nucl_only=nucl_only,
    )

    col_factors = quantile_init(
        array_row=array_cols,
        n_counts=n_counts,
        quant_init=quant_init,
        rows_idx_opt=cols_idx_opt,
        epsilon=epsilon,
        nucl_selector=nucl_selector,
        nucl_only=nucl_only,
    )
    return row_factors, col_factors


quantile_init_from_sdata_wo_zeros = two_stage_nonzero_factors_filter(
    quantile_init_from_sdata
)


def init_from_sdata_with_fun(
    data: spatialAdata,
    fun: callable,
    rows_idx_opt=None,
    cols_idx_opt=None,
    epsilon=0,
    nucl_only=False,
    cell_id_label="cell_id",
):
    """
    Adapter function to format input from a spatialAdata object for quantile_init.

    Parameters:
    - data (spatialAdata): The spatialAdata object containing the data.
    - quant_init (float): Quantile value to initialize the factors.
    - rows_idx_opt (np.ndarray, optional): Indices of rows to normalize. Default is None.
    - cols_idx_opt (np.ndarray, optional): Indices of columns to normalize. Default is None.
    - epsilon (float, optional): Minimum value for the factors. Default is 0.
    - nucl_only (bool, optional): If True, compute quantiles only on nuclear data. Default is False.
    - cell_id_label (str, optional): Label for cell IDs in the data. Default is "cell_id".

    Returns:
    - h (pd.Series): Row factors as a pandas Series, with other factors set to one.
    - w (pd.Series): Column factors as a pandas Series, with other factors set to one.
    """
    # Add row and column coordinates to the data
    data.n_counts
    data.add_array_coords_to_obs()

    # Extract necessary fields
    array_row = data.obs["array_row"].values
    array_cols = data.obs["array_col"].values
    n_counts = data.obs["n_counts"].values

    # Create a nuclear selector if needed
    nucl_selector = None
    if nucl_only:
        nucl_selector = ~data.obs[cell_id_label].isna().values

    # Call quantile_init with formatted inputs
    row_factors = custom_fun_init(
        array_row=array_row,
        n_counts=n_counts,
        fun=fun,
        rows_idx_opt=rows_idx_opt,
        epsilon=epsilon,
        nucl_selector=nucl_selector,
        nucl_only=nucl_only,
    )

    col_factors = custom_fun_init(
        array_row=array_cols,
        n_counts=n_counts,
        fun=fun,
        rows_idx_opt=cols_idx_opt,
        epsilon=epsilon,
        nucl_selector=nucl_selector,
        nucl_only=nucl_only,
    )
    return row_factors, col_factors


def ones_init_from_sdata(data: spatialAdata):
    # Add row and column coordinates to the data
    data.add_array_coords_to_obs()

    # Extract necessary fields
    array_row = data.obs["array_row"].values
    array_cols = data.obs["array_col"].values

    return ones_init(array_row), ones_init(array_cols)


def ratio_init_from_sdata(
    data: spatialAdata,
    rows_idx_opt=None,
    cols_idx_opt=None,
    epsilon=0,
    cell_id_label="cell_id",
    quantile=0.5,
    min_cell_bin_count=10,
):
    data.n_counts
    data.add_array_coords_to_obs()
    data_obs = data.obs.copy()

    nucl_indices = data.nucl_indices(cell_id_label)
    cell_median = data_obs.groupby(cell_id_label, observed=True)["n_counts"].median()
    cell_counts = data_obs.groupby(cell_id_label, observed=True)["n_counts"].count()

    cell_median.loc[cell_counts < min_cell_bin_count] = np.nan

    data_obs["cell_median"] = cell_median.reindex(data_obs[cell_id_label].values).values
    data_obs["ratio"] = data_obs.eval("n_counts/cell_median")

    results_dict = {}
    for factor_name, lane_name, idx_opt in [
        ("h", "row", rows_idx_opt),
        ("w", "col", cols_idx_opt),
    ]:
        factor = (
            data_obs.loc[nucl_indices]
            .groupby(f"array_{lane_name}")["ratio"]
            .quantile(q=quantile)
        )

        if idx_opt is not None:
            # if some rows_idx are missing, fill them with ones (this can happen for example if we do quantiles_nucl init and rows_idx_opt contain rows wo any nucl)
            factor_opt = factor.get(idx_opt, default=1.0)
            # h_opt = h[rows_idx_opt].copy()
        else:
            factor_opt = factor.copy()
        factor_opt /= np.mean(factor_opt)
        factor[:] = 1  # Set all factors to 1
        factor.update(factor_opt)  # Update only the selected indices
        factor = factor.reindex(data_obs[f"array_{lane_name}"].unique()).fillna(
            1
        )  # Reindex on original data
        factor = factor.clip(lower=epsilon)
        results_dict[factor_name] = factor

    return results_dict["h"], results_dict["w"]


ratio_init_from_sdata_wo_zeros = two_stage_nonzero_factors_filter(ratio_init_from_sdata)
