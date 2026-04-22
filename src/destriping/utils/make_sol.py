from src.destriping.sol import Sol
from src.spatialAdata.from_numpy import make_sdata
import numpy as np
import pandas as pd


def make_stripes_serie(lane_idx, stripe):
    return pd.Series(index=lane_idx, data=stripe)


def make_sol(
    h_row,
    h_values,
    w_col,
    w_values,
    data_row_indices,
    data_col_indices,
    data_index,
    fitted_c,
):
    h = make_stripes_serie(h_row, h_values)
    w = make_stripes_serie(w_col, w_values)
    h_per_bin = h.reindex(
        data_row_indices
    )  # used instead of .loc to avoid error in case of missing indices
    w_per_bin = w.reindex(data_col_indices)
    h_per_bin.fillna(1, inplace=True)
    w_per_bin.fillna(1, inplace=True)
    h_per_bin = h_per_bin.values
    w_per_bin = w_per_bin.values
    fitted_counts = h_per_bin * w_per_bin * fitted_c

    obs = {
        "fitted_c": fitted_c,
        "fitted_row_stripe_factor": h_per_bin,
        "fitted_col_stripe_factor": w_per_bin,
        "fitted_counts": fitted_counts,
    }
    obs = pd.DataFrame(index=data_index, data=obs)

    coordinates = np.array([data_row_indices, data_col_indices]).T  # nbins x 2 array
    count_matrix = fitted_c.reshape(-1, 1)
    var = pd.DataFrame(index=["tot_counts"])  # check this works
    c = make_sdata(coordinates, count_matrix, obs=obs, var=var)
    return Sol(h, w, c)


def make_lightweight_sol(h, w, data):
    return make_sol(
        h_row=h.index,
        h_values=h.values,
        w_col=w.index,
        w_values=w.values,
        data_row_indices=data.get_unscaled_coordinates("array")[:, 0],
        data_col_indices=data.get_unscaled_coordinates("array")[:, 1],
        data_index=data.index,
        fitted_c=data.n_counts,
    )
