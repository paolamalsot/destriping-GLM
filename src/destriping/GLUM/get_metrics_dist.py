from src.experiments_analysis.analysis_dist import distance_fun_dict, get_distance_fun
import warnings


def get_metrics_dist(fitted_sol, gt_sol, df):
    metrics = {}
    for sol in [fitted_sol, gt_sol]:
        sol.f = df["i"].map(sol.h).fillna(1.0) * df["j"].map(sol.w).fillna(1.0)

    for param_name in ["h", "w", "f"]:
        param = fitted_sol.__getattribute__(param_name)
        param_gt = gt_sol.__getattribute__(param_name)

        # make sure that all indices in gt are present -> fill with 1
        missing_indices = param_gt.index.difference(param.index)
        if len(missing_indices) > 0:
            warnings.warn(f"Metrics: {len(missing_indices)} missing indices in {param_name}. Filling with ones.")
        filled_param = param.reindex(param_gt.index).fillna(1.0)
        for metric_name, distance_fun in distance_fun_dict.items():
            distance_fun_ = get_distance_fun(metric_name, distance_fun, param_name)
            metrics[f"distance_to_gt_poisson_sol_{param_name}_{metric_name}"] = (
                distance_fun_(filled_param, param_gt)
            )

    metrics["distance_to_gt_poisson_sol_hw_euclidian"] = (
        metrics["distance_to_gt_poisson_sol_h_euclidian"] ** 2
        + metrics["distance_to_gt_poisson_sol_w_euclidian"] ** 2
    ) ** 0.5
    metrics["distance_to_gt_poisson_sol_hw_log_euclidian"] = (
        metrics["distance_to_gt_poisson_sol_h_log_euclidian"] ** 2
        + metrics["distance_to_gt_poisson_sol_w_log_euclidian"] ** 2
    ) ** 0.5
    return metrics