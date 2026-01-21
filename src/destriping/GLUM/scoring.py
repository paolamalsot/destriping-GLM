from glum._glm import get_family
import numpy as np
from src.destriping.GLUM.get_metrics_dist import get_metrics_dist
from src.destriping.GLUM.sol import mu_from_sol
from scipy.spatial.distance import cosine, euclidean

def get_score_from_sol(sol, df, family):
    # inspired from from glum._distribution import ExponentialDispersionModel, deviance
    family = get_family(family)
    y = df["k"].values
    mu = mu_from_sol(sol, df).values #was prev get_mu_from_sol
    dev = family.deviance(y, mu)
    y_mean = np.average(y)
    dev_null = family.deviance(y, y_mean)
    return 1.0 - dev / dev_null

def eval_glm(glm, fitted_sol, gt_sol, df):
    # returns a dict with the distance metrics, the score (deviance), the number of iterations
    if gt_sol is not None:
        dist_metrics = get_metrics_dist(fitted_sol, gt_sol, df)
    else:
        dist_metrics = {}
    score = get_score_from_sol(fitted_sol, df, glm.family)
    #global_structure_statistics_dict = global_structure_statistics(fitted_sol).to_dict()
    n_iters = glm.n_iter_
    metrics = {
        "score": score,
        "n_iters": n_iters,
        **dist_metrics,
        #**global_structure_statistics_dict,
    }
    return metrics


def normalized_euclidian_distance_fun(gt, v_j):
    intersect_indices = gt.index
    assert gt.values.ndim == 1
    dist_ = euclidean(
        gt[intersect_indices].values, v_j[intersect_indices].values
    ) / np.sqrt(
        len(intersect_indices)
    )  # euclidean is necessarily 1D
    return dist_


def normalized_euclidian_log_distance_fun(gt, v_j, offset):
    v_i_log = np.log(gt + offset)
    v_j_log = np.log(v_j + offset)
    dist_ = normalized_euclidian_distance_fun(v_i_log, v_j_log)
    return dist_


def cosine_distance_fun(gt, v_j):
    intersect_indices = gt.index
    dist_ = cosine(gt[intersect_indices].values, v_j[intersect_indices].values)
    return dist_


distance_fun_dict = {
    "cosine": cosine_distance_fun,
    "euclidian": normalized_euclidian_distance_fun,
    "log_euclidian": normalized_euclidian_log_distance_fun,
}
log_offset_dict = {"w": 1e-20, "h": 1e-20, "f": 1e-20, "n_counts_c": 1}
