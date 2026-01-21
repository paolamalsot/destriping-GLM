from scipy.spatial.distance import cosine, euclidean
import pandas as pd
import numpy as np


def normalized_euclidian_distance_fun_same_index(v_i, v_j):
    pd.testing.assert_index_equal(v_i.index, v_j.index)
    assert v_i.values.ndim == 1
    dist_ = euclidean(v_i.values, v_j.values) / np.sqrt(
        len(v_i.index)
    )  # euclidean is necessarily 1D
    return dist_


def normalized_euclidian_log_distance_fun_same_index(v_i, v_j, offset):
    v_i_log = np.log(v_i + offset)
    v_j_log = np.log(v_j + offset)
    dist_ = normalized_euclidian_distance_fun_same_index(v_i_log, v_j_log)
    return dist_


def cosine_distance_fun_same_index(v_i, v_j):
    pd.testing.assert_index_equal(v_i.index, v_j.index)
    dist_ = cosine(v_i.values, v_j.values)
    return dist_


log_offset_dict = {"w": 1e-20, "h": 1e-20, "f": 1e-20, "n_counts_c": 1}
distance_fun_dict_same_index = {
    "cosine": cosine_distance_fun_same_index,
    "euclidian": normalized_euclidian_distance_fun_same_index,
    "log_euclidian": normalized_euclidian_log_distance_fun_same_index,
}


def normalized_euclidian_distance_fun(v_i, v_j):
    intersect_indices = v_i.index.intersection(v_j.index)
    assert v_i.values.ndim == 1
    dist_ = euclidean(
        v_i[intersect_indices].values, v_j[intersect_indices].values
    ) / np.sqrt(
        len(intersect_indices)
    )  # euclidean is necessarily 1D
    return dist_


def normalized_euclidian_log_distance_fun(v_i, v_j, offset):
    v_i_log = np.log(v_i + offset)
    v_j_log = np.log(v_j + offset)
    dist_ = normalized_euclidian_distance_fun(v_i_log, v_j_log)
    return dist_


def cosine_distance_fun(v_i, v_j):
    intersect_indices = v_i.index.intersection(v_j.index)
    dist_ = cosine(v_i[intersect_indices].values, v_j[intersect_indices].values)
    return dist_


distance_fun_dict = {
    "cosine": cosine_distance_fun,
    "euclidian": normalized_euclidian_distance_fun,
    "log_euclidian": normalized_euclidian_log_distance_fun,
}
log_offset_dict = {"w": 1e-20, "h": 1e-20, "f": 1e-20, "n_counts_c": 1}


def get_distance_fun(metric_name, distance_fun, param_name):
    if metric_name == "log_euclidian":
        distance_fun_ = lambda v_i, v_j: distance_fun(
            v_i, v_j, log_offset_dict[param_name]
        )
    else:
        distance_fun_ = distance_fun
    return distance_fun_
