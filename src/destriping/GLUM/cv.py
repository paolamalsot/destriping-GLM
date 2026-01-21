import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from src.destriping.GLUM.custom_regressors.helpers import RegressorTypeLike
from src.destriping.GLUM.custom_regressors.helpers import extract_intercept_coef
from src.destriping.GLUM.glum_nb_helpers import hash
from src.utilities.pandas import log_df
import logging
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
import logging
from copy import deepcopy
from src.destriping.GLUM.custom_regressors.warm_start_wrapper import WarmStartWrapper
from src.destriping.GLUM.custom_regressors.helpers import (
    remove_prefix_from_kwargs,
    wrap_dict_with_prefix,
)

from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

logger = logging.getLogger("GLUM CV")
logger = logging.getLogger("GLUM CV")

def _is_alpha_param_name(name: str) -> bool:
    """
    Return True if `name` is an 'alpha-like' parameter name.

    We treat as alpha any parameter whose *last* segment (split by '__')
    is literally 'alpha', e.g.:

        'alpha'                  -> True
        'regressor__alpha'       -> True
        'foo__bar__alpha'        -> True
        'something__l1_ratio'    -> False
    """
    return name.split("__")[-1] == "alpha"

def _start_params_key(params):
    list_start_params_key = [
        key for key in params.keys() if key.split("__")[-1] == "start_params"
    ]
    if len(list_start_params_key) == 0:
        raise ValueError("No start params key")
    elif len(list_start_params_key)>1:
        raise ValueError(f"Ambiguous start_params keys: {list_start_params_key}")
    else:
        return list_start_params_key[0]


def _alpha_group_key(params):
    """
    Build a 'group key' for parameters that share the same alpha path.

    We ignore ALL 'alpha-like' parameters (plain 'alpha' or nested
    names like 'regressor__alpha'): these define the regularization
    path dimension, not the group identity.

    Example
    -------
    params1 = {"regressor__alpha": 1.0, "l1_ratio": 0.2, "family": "poisson"}
    params2 = {"regressor__alpha": 0.1, "l1_ratio": 0.2, "family": "poisson"}

    Both share the same non-alpha settings, so:

        _alpha_group_key(params1) == _alpha_group_key(params2)
        == (("family", "poisson"), ("l1_ratio", 0.2))

    Implementation details
    ----------------------
    - Drop all keys whose last '__'-segment is 'alpha'.
    - Turn remaining items into (key, value) pairs.
    - Sort these pairs by key so the order is stable and hashable.
    - Wrap in a tuple so we can use it as a dict key.
    """
    return tuple(
        sorted((k, v) for k, v in params.items() if not _is_alpha_param_name(k))
    )


def get_alpha_param_name(list_keys):
    alpha_keys = [k for k in list_keys if _is_alpha_param_name(k)]

    if not alpha_keys:
        raise KeyError(
            "No alpha-like parameter found in params; "
            "this should not happen when warm_start_alpha=True."
        )

    if len(alpha_keys) > 1:
        raise ValueError(
            f"Multiple alpha-like parameters found in params: {alpha_keys}. "
            "warm_start_alpha currently supports only a single alpha parameter "
            "(e.g. 'alpha' or 'regressor__alpha')."
        )

    return alpha_keys[0]


def _get_alpha_value(params):
    """
    Extract the alpha value from a parameter dict.

    We look for all 'alpha-like' keys (last '__'-segment is 'alpha').
    - If none are found: raise (this should not happen when warm_start_alpha is used).
    - If more than one is found: raise, since warm_start_alpha currently
      supports only a single alpha dimension.

    This keeps behaviour explicit and avoids ambiguous paths.
    """
    alpha_key = get_alpha_param_name(list(params.keys()))
    return params[alpha_key]


def _order_param_grid_for_alpha_path(param_grid_sk):
    """
    Reorder a list of param dicts so that, within each alpha group,
    we visit alphas from largest to smallest.

    Example
    -------
    Given the raw (unsorted) parameter list:

        [
            {"alpha": 0.1, "l1_ratio": 0.2},
            {"alpha": 1.0, "l1_ratio": 0.8},
            {"alpha": 0.5, "l1_ratio": 0.2},
            {"alpha": 0.1, "l1_ratio": 0.8},
            {"alpha": 1.0, "l1_ratio": 0.2},
        ]

    The groups (by non-alpha params) are:

        Group A: l1_ratio = 0.2
            {"alpha": 1.0, "l1_ratio": 0.2}
            {"alpha": 0.5, "l1_ratio": 0.2}
            {"alpha": 0.1, "l1_ratio": 0.2}

        Group B: l1_ratio = 0.8
            {"alpha": 1.0, "l1_ratio": 0.8}
            {"alpha": 0.1, "l1_ratio": 0.8}

    This function returns them ordered as:

        [
            {"alpha": 1.0, "l1_ratio": 0.2},
            {"alpha": 0.5, "l1_ratio": 0.2},
            {"alpha": 0.1, "l1_ratio": 0.2},

            {"alpha": 1.0, "l1_ratio": 0.8},
            {"alpha": 0.1, "l1_ratio": 0.8},
        ]

    i.e. for each group, alphas go from large → small, which is what we want
    for warm-starting along a regularization path.


    Reorder a list of param dicts so that, within each alpha group
    (same non-alpha params), we visit alphas from largest to smallest.

    Works with both plain 'alpha' and nested names like 'regressor__alpha'.
    """
    return sorted(
        param_grid_sk,
        key=lambda p: (_alpha_group_key(p), -_get_alpha_value(p)),
    )

def get_alpha_path(param_grid_sk, best_params, final_alpha_val):
    # we suppose param_grid_sk is already ordered
    # best_params contain the best_params not the whole glm_params
    alphas = []
    for combi in param_grid_sk:
        if _alpha_group_key(combi) == _alpha_group_key(best_params):
            alpha = _get_alpha_value(combi)
            if alpha < final_alpha_val:
                break
            alphas.append(alpha)

    # check that alphas are sorted from big to small and that the last value is final_alpha_val
    assert sorted(alphas)[::-1] == alphas
    assert alphas[-1] == final_alpha_val
    return alphas


def set_start_params(glm_kwargs, params, i_split, warm_starts):
    """
    Return glm_kwargs for this (params, fold) with an appropriate start_params.

    - Always returns a *copy* of glm_kwargs (does not modify in place).
    - Assumes warm_starts is a dict.
    - Looks for a stored warm start for this (group, fold) and, if found,
      sets glm_kwargs_split["start_params"] accordingly.

    This means:
    - For the *first* alpha in a given (group, fold), there is no entry yet
      in warm_starts, so any start_params/start_coef you had in the original
      glm_kwargs are used as-is.
    - For subsequent alphas in the same (group, fold), we overwrite
      start_params with the previously stored coefficients.
    """
    glm_kwargs_split = deepcopy(glm_kwargs)  # shallow copy

    group_key = _alpha_group_key(params)
    ws_key = (group_key, i_split)
    start = warm_starts.get(ws_key)

    if start is not None:
        start_params_key = _start_params_key(glm_kwargs_split)
        glm_kwargs_split[start_params_key] = start
        logger.debug(
            f"Warm_start for group {group_key}, and split {i_split} (Hash: {hash(start)})"
        )

    return glm_kwargs_split


def store_coef(glm, params, i_split, warm_starts):
    """
    Store the fitted coefficients as warm start for the next alpha
    for this (group, fold).

    Assumes warm_starts is a dict.

    Stored format matches glum's 'start_params' convention:
        [intercept, coef_0, coef_1, ..., coef_p]

    If fit_intercept is False, we only store the coefficients.
    """
    group_key = _alpha_group_key(params)
    ws_key = (group_key, i_split)

    start_params_next = extract_intercept_coef(glm)

    warm_starts[ws_key] = start_params_next
    logger.debug(f"Storing param (hash = {hash(start_params_next)} for group {ws_key})")

def alpha_in_param_grid(param_grid):
    alpha_keys_in_grid = [k for k in param_grid.keys() if _is_alpha_param_name(k)]
    if not alpha_keys_in_grid:
        return False  # nothing to warm-start on
    elif len(alpha_keys_in_grid) > 1:
        raise ValueError(
            f"warm_start_alpha=True but param_grid contains multiple "
            f"alpha-like parameters: {alpha_keys_in_grid}. "
            "Currently only a single alpha dimension is supported."
        )
    else:
        return True

def get_warm_start_regressor(regressor_class, base_kwargs, best_params, param_grid_sk):

    alpha_param_key = get_alpha_param_name(list(best_params.keys()))
    final_alpha_val = best_params[alpha_param_key]

    alpha_path = get_alpha_path(
        param_grid_sk, best_params, final_alpha_val
    )

    start_params_key = _start_params_key(base_kwargs)

    best_glm_kwargs_wo_alpha = {**base_kwargs, **best_params}
    del best_glm_kwargs_wo_alpha[alpha_param_key]

    warm_start_regressor = WarmStartWrapper(
        regressor_class,
        alpha_param_key,
        start_params_key,
        alpha_path,
        **wrap_dict_with_prefix(best_glm_kwargs_wo_alpha, "regressor"),
    )

    return warm_start_regressor


## MAIN LOGIC

def _run_one_split_df(
    *,
    i_split: int,
    train_idx,
    test_idx,
    X,
    y,
    offset,
    sample_weight,
    base_kwargs: dict,
    param_grid_sk: list[dict],
    regressor_class,
    use_alpha_warm_start: bool,
):
    """
    Run ALL parameter settings for a single split, sequentially.
    Returns a DataFrame with one row per (param_i, split).
    """
    logger.debug(f"CV(split={i_split}): slicing data")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if offset is not None:
        offset_train, offset_test = offset[train_idx], offset[test_idx]
    else:
        offset_train = offset_test = None

    if sample_weight is not None:
        w_train, w_test = sample_weight[train_idx], sample_weight[test_idx]
    else:
        w_train = w_test = None

    # Important: warm starts must be local to this split/job.
    warm_starts = {} if use_alpha_warm_start else None

    rows = []
    fold_len = len(y_test)

    for param_i, params in enumerate(param_grid_sk):
        glm_kwargs = {**base_kwargs, **params}

        if use_alpha_warm_start:
            glm_kwargs_split = set_start_params(
                glm_kwargs=glm_kwargs,
                params=params,
                i_split=i_split,
                warm_starts=warm_starts,
            )
        else:
            glm_kwargs_split = dict(glm_kwargs)

        glm = regressor_class(**glm_kwargs_split)
        glm.fit(X_train, y_train, sample_weight=w_train, offset=offset_train)

        if use_alpha_warm_start:
            store_coef(glm=glm, params=params, i_split=i_split, warm_starts=warm_starts)

        mu = glm.predict(X_test, offset=offset_test)
        dev = glm._family_instance.deviance(y_test, mu, sample_weight=w_test)

        coef_sparsity = int(np.count_nonzero(glm.coef_ == 0))
        coef_norm = float(np.linalg.norm(glm.coef_, ord=2))

        rows.append(
            {
                "param_i": param_i,
                "split": i_split,
                "fold_length": fold_len,
                "dev": float(dev),
                "coef_sparsity": coef_sparsity,
                "coef_norm": coef_norm,
            }
        )

    return pd.DataFrame(rows)


def glm_cv(
    X,
    y,
    arguments,
    param_grid,
    *,
    offset=None,
    sample_weight=None,
    one_SE_rule=False,
    regressor_class: RegressorTypeLike,
    warm_start_alpha: bool = False,
    parallel: bool = True,
    n_jobs: int = -1,
):
    """ Manual CV for glum.GeneralizedLinearRegressor with deviance scoring. 
    Parameters
        X : dataframe-like, shape (n_samples, n_features) 
        y : array-like, shape (n_samples,) 
        arguments : dict Base keyword arguments for GeneralizedLinearRegressor. Must contain a key "cv" with either: - an sklearn splitter (KFold, etc.) OR - an iterable of (train_idx, test_idx) tuples param_grid : dict Hyper-parameter grid, sklearn-style, e.g. {"alpha": [0.01, 0.1], "l1_ratio": [0, 0.5]} 
        offset : array-like, shape (n_samples,), optional 
        sample_weight : array-like, shape (n_samples,), optional 
        one_SE_rule: whether to choose the best hp with one_SE_rule
        regressor_class: class of regressor
        warm_start_alpha: boolean flag for performing warm starts from higher alpha values 
        parallel: boolean flag for performing parallel fits among splits. 
    Returns ------- 
        best_estimator : object of type regressor_class 
        best_params : dict 
        best_score : float Mean deviance over folds for best_params (lower is better). 
        cv_results : list of dict One dict per parameter setting with mean/fold deviances, etc. 

    Parallelization strategy:
    - parallel=True  => parallel over splits (joblib), sequential over params per split
    - parallel=False => fully sequential

    warm_start_alpha remains valid because each split is evaluated sequentially
    over the ordered alpha path inside its own job.
    """

    if "cv" not in arguments:
        raise ValueError("arguments must contain a 'cv' key with the cross-validation splits")

    cv = arguments["cv"]
    base_kwargs = {k: v for k, v in arguments.items() if k != "cv"}

    # Turn cv into a list of (train_idx, test_idx)
    if hasattr(cv, "split"):
        splits = list(cv.split(X, y))
    else:
        splits = list(cv)

    param_grid_sk = list(ParameterGrid(param_grid))

    use_alpha_warm_start = warm_start_alpha and alpha_in_param_grid(param_grid)
    param_grid_sk = _order_param_grid_for_alpha_path(param_grid_sk)

    logger.debug(
        f"Launching CV for regressor class {regressor_class.__name__} "
        f"and param_grid {param_grid} and base-params {base_kwargs}: "
        f"parallel={parallel}, n_jobs={n_jobs}, warm_start_alpha={warm_start_alpha}"
    )

    # --- Run splits (parallel or sequential), each returns a df ---
    if parallel:
        split_dfs = Parallel(n_jobs=n_jobs)(
            delayed(_run_one_split_df)(
                i_split=i_split,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                offset=offset,
                sample_weight=sample_weight,
                base_kwargs=base_kwargs,
                param_grid_sk=param_grid_sk,
                regressor_class=regressor_class,
                use_alpha_warm_start=use_alpha_warm_start,
            )
            for i_split, (train_idx, test_idx) in enumerate(splits)
        )
    else:
        split_dfs = [
            _run_one_split_df(
                i_split=i_split,
                train_idx=train_idx,
                test_idx=test_idx,
                X=X,
                y=y,
                offset=offset,
                sample_weight=sample_weight,
                base_kwargs=base_kwargs,
                param_grid_sk=param_grid_sk,
                regressor_class=regressor_class,
                use_alpha_warm_start=use_alpha_warm_start,
            )
            for i_split, (train_idx, test_idx) in enumerate(splits)
        ]

    df_long = pd.concat(split_dfs, ignore_index=True)
    
    df_params = pd.DataFrame({"param_i": np.arange(len(param_grid_sk)), "params": param_grid_sk})
    df_long = df_long.merge(df_params, on="param_i", how="left")

    # --- Groupby gymnastics to build df_results ---
    k = len(splits)

    def _agg_one_param(g: pd.DataFrame) -> pd.Series:
        # Weighted mean deviance (sum dev / sum fold_length), matching your original code.
        total_len = float(g["fold_length"].sum())
        mean_unit_dev = float(g["dev"].sum()) / total_len

        # Weighted variance of per-unit deviance
        per_unit = g["dev"].to_numpy() / g["fold_length"].to_numpy()
        w = g["fold_length"].to_numpy().astype(float)
        var_dev = float(np.sum(w * (per_unit - mean_unit_dev) ** 2) / np.sum(w))

        se_dev = float(np.sqrt(var_dev / k))

        return pd.Series(
            {
                "params": g["params"].iloc[0],
                "mean_deviance": mean_unit_dev,
                "se_deviance": se_dev,
                "fold_deviances": list(g.sort_values("split")["dev"].astype(float).to_list()),
                "coef_sparsity": float(g["coef_sparsity"].mean()),
                "coef_norm": float(g["coef_norm"].mean()),
            }
        )

    df_results = df_long.groupby("param_i", sort=True).apply(_agg_one_param).reset_index(drop=True)

    logger.debug("Finished CV")
    log_df(df_results, logger.debug)

    # --- Selection logic (unchanged) ---
    best_index = df_results["mean_deviance"].idxmin()
    min_deviance = df_results["mean_deviance"].min()
    se_best_index = df_results.at[best_index, "se_deviance"]
    sparsity_best_index = df_results.at[best_index, "coef_sparsity"]
    coef_norm_best_index = df_results.at[best_index, "coef_norm"]

    logger.debug(
        f"Best index {best_index}: deviance = {min_deviance} pm {se_best_index}, "
        f"sparsity = {sparsity_best_index}, coef_norm = {coef_norm_best_index}"
    )

    if one_SE_rule:
        logger.debug("Selecting the best HP according to one_SE_rule")
        within_1_SE = df_results.query("mean_deviance < @min_deviance + @se_best_index")
        logger.debug(
            f"Considering {len(within_1_SE)} other parameters combinations that are within 1 SE"
        )
        sparsest_index = within_1_SE["coef_sparsity"].idxmax()
        sparsity_max = df_results.at[sparsest_index, "coef_sparsity"]
        if sparsity_max > sparsity_best_index:
            logger.debug(
                f"Selecting the HP because of higher sparsity ({sparsity_max=} > {sparsity_best_index=})"
            )
            new_best_index = sparsest_index
        else:  # select smallest norm
            smallest_norm_index = within_1_SE["coef_norm"].idxmin()
            smallest_norm = within_1_SE["coef_norm"].min()
            if smallest_norm < coef_norm_best_index:
                logger.debug(
                    f"Selecting the HP because of smaller norm ({smallest_norm=} < {coef_norm_best_index=})"
                )
                new_best_index = smallest_norm_index
            else:
                logger.debug(
                    f"Selecting the best hp because of equality of norm ({smallest_norm=}, {coef_norm_best_index=}) "
                    f"and sparsity ({sparsity_max=} = {sparsity_best_index=}) within 1-SE."
                )
                new_best_index = best_index
        best_index = new_best_index

    best_glm_params = df_results.at[best_index, "params"]
    best_score = df_results.at[best_index, "mean_deviance"]
    best_glm_kwargs = {**base_kwargs, **best_glm_params}

    # Refit on full data with best params
    logger.debug(f"Best parameters: {best_glm_params}")
    logger.debug("Refit best estimator on full-training data")

    if use_alpha_warm_start:
        best_glm = get_warm_start_regressor(
            regressor_class, base_kwargs, best_glm_params, param_grid_sk
        )
        best_glm_params["_alpha_path"] = best_glm.alpha_path
    else:
        best_glm = regressor_class(**best_glm_kwargs)

    best_glm.fit(X, y, sample_weight=sample_weight, offset=offset)
    best_estimator = best_glm

    # Keep your original return type: `results` as list[dict]
    results = df_results.to_dict(orient="records")
    return best_estimator, best_glm_params, best_score, results