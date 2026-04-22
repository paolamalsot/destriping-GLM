import functools
import numpy as np
from warnings import warn
import numpy as np
import pandas as pd
from src.destriping.GLUM.iterative_theta import theta_cal
from typing import Callable
from src.destriping.GLUM.custom_regressors.helpers import wrap_dict_with_prefix
from src.destriping.GLUM.custom_regressors.cv_regressor import CustomCVRegressor
from src.destriping.GLUM.custom_regressors.iterative_theta_regressor import (
    IterativeThetaGLM,
)
from src.destriping.GLUM.custom_regressors.alternating_theta_cv_regressor import (
    AlternatingThetaCVRegressor,
)

from warnings import warn
from src.destriping.GLUM.glum_nb_helpers import family_is_negative_binomial
import pandas as pd
from pandas.api.types import is_string_dtype
from types import SimpleNamespace
import logging
from src.destriping.GLUM.custom_regressors.cv_regressor import CustomCVRegressor
from src.destriping.GLUM.custom_regressors.iterative_theta_after_cv_regressor import (
    IterativeThetaAfterCVRegressor,
)
from src.destriping.GLUM.custom_regressors.iterative_theta_regressor import (
    IterativeThetaGLM,
)
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV
from dask_ml.preprocessing import Categorizer
import tabmat as tm
from src.destriping.GLUM.coordinate_descent_solver import (
    coordinate_descent_lbfgs_solver,
)

logger = logging.getLogger("Glum fit")


def extract_categories_dict_from_categorizer(ce):
    categories_dict = {
        col: dtype.categories.tolist() for col, dtype in ce.categories_.items()
    }
    return categories_dict


def pop_param_grid_from_GLM_args(glm_arguments):
    param_grid = {
        "alpha": glm_arguments["alphas"]
    }  # Note that in GeneralizedLinearRegressor, alphas is deprec.
    del glm_arguments["alphas"]
    return param_grid


def _compute_theta_init(
    h_start: float | None,
    w_start: float | None,
    c_start: float | None,
    theta_cal: Callable,
    df,
) -> float:
    if h_start is None or w_start is None or c_start is None:
        return 1.0

    init_sol = SimpleNamespace(h=h_start, w=w_start, c=c_start)
    return theta_cal(init_sol, df)


def factor_to_glum_coef(start, levels, name_level, epsilon):
    # warnings about missing levels
    missing_levels = list(set(levels).difference(set(start.index.tolist())))
    if len(missing_levels) > 0:
        warn(f"In category {name_level}, missing levels: {missing_levels}")

    if (start < epsilon).any():
        n_vals = (start < epsilon).sum()
        warn(
            f"In category {name_level}, {n_vals} values smaller than epsilon ({epsilon}). Clipping."
        )

    coef_with_dropped = (
        start.reindex(levels).fillna(1).clip(lower=epsilon).apply(np.log)
    )
    dropped_level_name = levels[0]
    intercept_contribution = coef_with_dropped.iloc[0]
    coef_contribution = coef_with_dropped.iloc[1:] - intercept_contribution

    features_names_contribution = (
        coef_contribution.index.to_series()
        .apply(lambda x: name_level + "[" + str(x) + "]")
        .values.tolist()
    )
    return (
        coef_contribution,
        intercept_contribution,
        features_names_contribution,
        dropped_level_name,
    )


def h_w_c_to_glum_coef(
    h: pd.Series,
    w: pd.Series,
    c: pd.Series,
    levels_i: list[str],
    levels_j: list[str],
    levels_p: list[str],
    epsilon: float = 1e-12,
):
    # Convert the h,w,c to the glum_coef such that the underlying model for the mean is h * w * c.
    # We assume that the glum has a log-link with drop_first = True and fit_intercept = True
    # levels_i, levels_j, levels_p are the levels expected by the glum-model (including the dropped category)
    # epsilon is the minimum value for h, w, c, otherwise the log is not defined.
    # missing coefs are replaced by zero.

    # returns:
    #   - np.array with concatenated intercept and glum_coef corresponding to the levels with the first category dropped
    #   - list with the feature names corresponding to the coefficients with the categorical-format '{name}[{category}]'

    # initial checks
    # assert that h, w integer indices.
    assert h.index.dtype == int
    assert w.index.dtype == int
    # assert that c have str indices
    assert is_string_dtype(c.index)

    intercept = 0
    coef = [0]  # for 0 intercept
    features_names = []
    dropped_levels_dict = {}

    for name_level, levels, start in [
        ("p", levels_p, c),
        ("i", levels_i, h),
        ("j", levels_j, w),
    ]:
        (
            coef_contribution,
            intercept_contribution,
            features_names_contribution,
            dropped_level_name,
        ) = factor_to_glum_coef(start, levels, name_level, epsilon)

        intercept += intercept_contribution
        coef.extend(coef_contribution.values)
        features_names.extend(features_names_contribution)
        dropped_levels_dict[name_level] = dropped_level_name

    coef[0] = intercept
    return np.array(coef), features_names, dropped_levels_dict


def c_to_offset(c, cell_id, epsilon: float = 1e-12):
    # cell_id: pd.Series of length n_obs
    # c: pd.Series with indices cell_id and values c
    # returns log(c[cell_id]) np.array of length n_obs

    if (c < epsilon).any():
        n_vals = (c < epsilon).sum()
        warn(f"In c, {n_vals} values smaller than epsilon ({epsilon}). Clipping.")

    # warnings about missing levels in c
    missing_levels = list(set(cell_id.values).difference(set(c.index.tolist())))
    if len(missing_levels) > 0:
        warn(f"In c, missing levels: {missing_levels}")

    c_log = c.reindex(cell_id.values).fillna(1).clip(lower=epsilon).apply(np.log).values
    return c_log


def h_w_to_glum_coef(
    h: pd.Series,
    w: pd.Series,
    levels_i: list[str],
    levels_j: list[str],
    epsilon: float = 1e-12,
):
    # same as h_w_c_to_glum_coef but without c

    # initial checks
    # assert that h, w integer indices.
    assert h.index.dtype == int
    assert w.index.dtype == int

    intercept = 0
    coef = [0]  # for 0 intercept
    features_names = []
    dropped_levels_dict = {}

    for name_level, levels, start in [
        ("i", levels_i, h),
        ("j", levels_j, w),
    ]:
        (
            coef_contribution,
            intercept_contribution,
            features_names_contribution,
            dropped_level_name,
        ) = factor_to_glum_coef(start, levels, name_level, epsilon)

        intercept += intercept_contribution
        coef.extend(coef_contribution.values)
        features_names.extend(features_names_contribution)
        dropped_levels_dict[name_level] = dropped_level_name

    coef[0] = intercept
    return np.array(coef), features_names, dropped_levels_dict


@functools.lru_cache(maxsize=4)
def _parse_feature_names(feature_names_tuple):
    """Parse 'name[category]' feature names into per-variable index.

    Returns dict: var_name -> (np.array of indices, list of categories).
    Cached by feature_names content (caller must pass a tuple).
    """
    var_indices = {}
    for i, fn in enumerate(feature_names_tuple):
        bracket = fn.index("[")
        name = fn[:bracket]
        category = fn[bracket + 1 : -1]
        if name not in var_indices:
            var_indices[name] = ([], [])
        var_indices[name][0].append(i)
        var_indices[name][1].append(category)
    return {
        name: (np.array(indices), cats)
        for name, (indices, cats) in var_indices.items()
    }


def extract_coef(coef, feature_names):
    # coef is a numpy array with same size as feature_names
    # feature_names are glm.feature_names with the categorical-format '{name}[{category}]'
    # returns a pandas df with columns name, column category and corresponding value
    assert len(coef) == len(feature_names)
    var_indices = _parse_feature_names(tuple(feature_names))
    names = np.empty(len(feature_names), dtype=object)
    categories = np.empty(len(feature_names), dtype=object)
    for var_name, (indices, cats) in var_indices.items():
        names[indices] = var_name
        categories[indices] = cats
    return pd.DataFrame(
        {"feature_names": feature_names, "coef": coef, "name": names, "category": categories}
    )


def extract_coef_specific(coef, feature_names, var_name):
    var_indices = _parse_feature_names(tuple(feature_names))
    if var_name not in var_indices:
        return pd.Series(dtype=float, name="coef")
    indices, categories = var_indices[var_name]
    return pd.Series(coef[indices], index=categories, name="coef")


def rescale_hwc(h: pd.Series, w: pd.Series, c: pd.Series, exp_intercept: float):
    # Rescale the solutions h, w, c, exp_intercept such that we get an equivalent model mu[ijp] = h_s[i] * w_s[j] * c_s[p]
    # with sum(h_s) = len(h_s) and sum(w_s) = len(w_s)
    f_h = len(h) / np.sum(h)
    f_w = len(w) / np.sum(w)
    f_c = exp_intercept / (f_h * f_w)
    return h * f_h, w * f_w, c * f_c

UPPER_EXP_CLIP = 30

def extract_coef_with_dropped_level(
    glum_coef, glum_feature_names, coef_key, dropped_level
):
    coef_specific = extract_coef_specific(glum_coef, glum_feature_names, coef_key)
    assert not (dropped_level in coef_specific.index)
    coef_specific.loc[dropped_level] = 0.0
    coef_specific = coef_specific.clip(upper=UPPER_EXP_CLIP).apply(np.exp)
    return coef_specific


def glum_coef_to_hwc(coef, intercept, feature_names, dropped_levels_dict):
    # Convert the glum's coef and intercept into h,w,c such that the underlying model for the mean is h * w * c.
    # We assume that the coef comes from a glum with a log-link with drop_first = True and fit_intercept = True
    # feature_names are glm.feature_names with the categorical-format '{name}[{category}]'
    # dropped_levels_dict has keys i, j, p and corresponding
    # returns h, w with indices as type int

    coef_dict = {}
    for coef_name, key in [("h", "i"), ("w", "j"), ("c", "p")]:
        dropped_level = dropped_levels_dict[key]
        coef_specific = extract_coef_with_dropped_level(
            coef, feature_names, key, dropped_level
        )
        coef_dict[coef_name] = coef_specific

    h, w, c = rescale_hwc(**coef_dict, exp_intercept=np.exp(np.clip(intercept, a_min = None, a_max=UPPER_EXP_CLIP)))
    h.index = h.index.to_numpy().astype(int)
    w.index = w.index.to_numpy().astype(int)
    return h, w, c


def glum_coef_to_hwc_frozen_c(
    coef, intercept, feature_names, frozen_c, dropped_levels_dict
):
    # for when c is not fit but used as offset
    # frozen_c is a pandas Serie
    coef_dict = {}
    for coef_name, key in [("h", "i"), ("w", "j")]:
        dropped_level = dropped_levels_dict[key]
        coef_specific = extract_coef_with_dropped_level(
            coef, feature_names, key, dropped_level
        )
        coef_dict[coef_name] = coef_specific

    coef_dict["c"] = frozen_c
    h, w, c = rescale_hwc(**coef_dict, exp_intercept=np.exp(intercept))
    h.index = h.index.to_numpy().astype(int)
    w.index = w.index.to_numpy().astype(int)
    return h, w, c


def _build_iterative_with_sklearn_cv(
    fit_theta_iter_loc: str,
    theta_init: float,
    fit_theta_max_iter: int,
    glm_arguments: dict,
    cv,
    warm_start_alpha,
    regressorCV_one_SE_rule: bool,
    parallel: bool,
    regressor_class=GeneralizedLinearRegressor,
    reestimate_c_on_val: bool = False,
):
    param_grid = pop_param_grid_from_GLM_args(glm_arguments)

    match fit_theta_iter_loc:
        case "in":
            inner_regressor_args = {
                "theta_max_iter": fit_theta_max_iter,
                "theta_init": theta_init,
                **wrap_dict_with_prefix(glm_arguments, "regressor"),
            }
            return CustomCVRegressor(
                wrap_dict_with_prefix(param_grid, "regressor"),
                one_SE_rule=regressorCV_one_SE_rule,
                regressor_class=IterativeThetaGLM,
                cv=cv,
                warm_start_alpha=warm_start_alpha,
                parallel=parallel,
                reestimate_c_on_val=reestimate_c_on_val,
                **wrap_dict_with_prefix(inner_regressor_args, "regressor"),
            )

        case "out":
            inner_regressor_args = {
                "param_grid": param_grid,
                "one_SE_rule": regressorCV_one_SE_rule,
                "regressor_class": regressor_class,
                "cv": cv,
                "warm_start_alpha": warm_start_alpha,
                "parallel": parallel,
                "reestimate_c_on_val": reestimate_c_on_val,
                **wrap_dict_with_prefix(glm_arguments, "regressor"),
            }
            inner_regressor_class = CustomCVRegressor
            return IterativeThetaGLM(
                theta_max_iter=fit_theta_max_iter,
                theta_init=theta_init,
                regressor_class=inner_regressor_class,
                family_arg_name="regressor__family",
                **wrap_dict_with_prefix(inner_regressor_args, "regressor"),
            )

        case "after":
            return IterativeThetaAfterCVRegressor(
                param_grid=param_grid,
                one_SE_rule=regressorCV_one_SE_rule,
                cv=cv,
                warm_start_alpha=warm_start_alpha,
                parallel=parallel,
                theta_max_iter=fit_theta_max_iter,
                theta_init=theta_init,
                regressor_class=regressor_class,
                **wrap_dict_with_prefix(glm_arguments, "regressor"),
            )

        case "alternating":
            return AlternatingThetaCVRegressor(
                param_grid=param_grid,
                one_SE_rule=regressorCV_one_SE_rule,
                cv=cv,
                warm_start_alpha=warm_start_alpha,
                parallel=parallel,
                theta_max_iter=fit_theta_max_iter,
                theta_init=theta_init,
                regressor_class=regressor_class,
                reestimate_c_on_val=reestimate_c_on_val,
                **wrap_dict_with_prefix(glm_arguments, "regressor"),
            )

        case _:
            raise ValueError(f"Unknown fit_theta_iter_loc: {fit_theta_iter_loc!r}")


def _build_iterative_regressor(
    sklearnCV: bool,
    fit_theta_iter_loc: str,
    theta_init: float,
    fit_theta_max_iter: int,
    glm_arguments: dict,
    cv,
    warm_start_alpha,
    parallel: bool,
    regressorCV_one_SE_rule: bool,
    regressor_class=GeneralizedLinearRegressor,
    reestimate_c_on_val: bool = False,
):
    if sklearnCV:
        return _build_iterative_with_sklearn_cv(
            fit_theta_iter_loc=fit_theta_iter_loc,
            theta_init=theta_init,
            fit_theta_max_iter=fit_theta_max_iter,
            glm_arguments=glm_arguments,
            cv=cv,
            warm_start_alpha=warm_start_alpha,
            parallel=parallel,
            regressorCV_one_SE_rule=regressorCV_one_SE_rule,
            regressor_class=regressor_class,
            reestimate_c_on_val=reestimate_c_on_val,
        )

    # fit_theta_iter and no sklearnCV
    return IterativeThetaGLM(
        theta_max_iter=fit_theta_max_iter,
        theta_init=theta_init,
        regressor_class=regressor_class,
        **wrap_dict_with_prefix(glm_arguments, "regressor"),
    )


def _build_non_iterative_regressor(
    regressorCV: bool,
    sklearnCV: bool,
    glm_arguments: dict,
    cv,
    warm_start_alpha,
    parallel,
    sklearnCV_one_SE_rule: bool,
    regressor_class=GeneralizedLinearRegressor,
    reestimate_c_on_val: bool = False,
):
    if regressorCV and sklearnCV:
        raise ValueError("regressorCV and sklearnCV are mutually exclusive")

    if regressorCV:
        if parallel:
            args_supp = {"n_jobs": -1}
        else:
            args_supp = {}
        return GeneralizedLinearRegressorCV(**glm_arguments, cv=cv, **args_supp)

    if sklearnCV:
        param_grid = pop_param_grid_from_GLM_args(glm_arguments)

        return CustomCVRegressor(
            param_grid,
            one_SE_rule=sklearnCV_one_SE_rule,
            regressor_class=regressor_class,
            cv=cv,
            parallel=parallel,
            warm_start_alpha=warm_start_alpha,
            reestimate_c_on_val=reestimate_c_on_val,
            **wrap_dict_with_prefix(glm_arguments, "regressor"),
        )

    return regressor_class(**glm_arguments)


def categorizer_glum(df, freeze_c):
    if freeze_c:
        categoricals = ["i", "j"]
    else:
        categoricals = ["p", "i", "j"]
    glm_categorizer = Categorizer(columns=categoricals)
    glm_categorizer.fit(df[categoricals])
    categories_dict = extract_categories_dict_from_categorizer(glm_categorizer)
    return categoricals, glm_categorizer, categories_dict


def fit_GLM_glum(
    df,
    h_start=None,
    w_start=None,
    c_start=None,
    freeze_c=False,
    regressorCV=False,
    sklearnCV=False,
    sklearnCV_one_SE_rule=True,
    cv=None,
    warm_start_alpha=False,
    parallel=False,
    fit_theta_iter=False,
    fit_theta_max_iter=3,
    fit_theta_iter_loc="in",
    solver="glum",
    reestimate_c_on_val=False,
    **kwargs,
):
    """Fit k_ij ~ NB( mean = c_p * h_i * w_j , var = mu + theta * mu^2 ) with a log link.

    Returns (h_hat, w_hat, glm)

    - df columns i, j can be either string, categories or integers
    - df columns p contain the cell id -> they must not contain any NAs !

    freeze_c: if True, we fit only the h and w coefficients. Note that the final c solution won't be necessarily the same, but a rescaled version to assure that sum(w)=n_cols and sum(h)=n_rows
    fit_theta_iter: False/True
    fit_theta_iter_loc: "in", "out", "after" -> has no influence if no CV specified or fit_theta_iter = False
    """

    link_spec = "log"

    # ---- determine regressor class based on solver --------------------------
    if solver in ("coordinate_descent", "lbfgs_control"):
        from src.destriping.GLUM.coordinate_descent_glm import (
            CoordinateDescentRegressor,
            LBFGSControlRegressor,
        )
        regressor_class = (
            CoordinateDescentRegressor if solver == "coordinate_descent"
            else LBFGSControlRegressor
        )
    else:
        regressor_class = GeneralizedLinearRegressor

    # ---- original glum path ----------------------------------

    # this step is equivalent to .astype(category), but the fitted categorizer allows to pass a correct df for an input that does not contain all categories ! Or will prob. raise an error if too many categories...
    categoricals, glm_categorizer, categories_dict = categorizer_glum(df, freeze_c)
    X = glm_categorizer.fit_transform(df[categoricals])
    y = df["k"].values

    # Save the full category sets so we can reconstruct effects in original order
    categories_dict = extract_categories_dict_from_categorizer(glm_categorizer)
    dropped_levels_dict = {key: cats[0] for key, cats in categories_dict.items()}

    # Warm-start
    if freeze_c:
        if c_start is None:
            raise ValueError("c_start must be provided when freeze_c = True.")
        offset = c_to_offset(c_start, df["p"])
    else:
        offset = None

    if freeze_c:
        if not (h_start is None) and not (w_start is None):
            (
                start_params,
                start_features_names,
                start_dropped_levels_dict,
            ) = h_w_to_glum_coef(
                h_start,
                w_start,
                levels_i=categories_dict["i"],
                levels_j=categories_dict["j"],
            )
        elif not (h_start is None) or not (w_start is None):
            start_params = None
            start_features_names = None
            warn("h_start w_start must be provided together otherwise not considered.")
        else:
            start_params = None
            start_features_names = None

    else:
        if not (h_start is None) and not (w_start is None) and not (c_start is None):
            # Build start_params from your h_hat, w_hat *and* a rough p median
            (
                start_params,
                start_features_names,
                start_dropped_levels_dict,
            ) = h_w_c_to_glum_coef(
                h_start,
                w_start,
                c_start,
                levels_i=categories_dict["i"],
                levels_j=categories_dict["j"],
                levels_p=categories_dict["p"],
            )
        elif not (h_start is None) or not (w_start is None) or not (c_start is None):
            start_params = None
            start_features_names = None
            warn(
                "h_start w_start and c_start must be provided together otherwise not considered."
            )
        else:
            start_params = None
            start_features_names = None

    # Fit GLM
    glm_arguments = {
        "link": link_spec,
        "fit_intercept": True,
        "drop_first": True,
        "start_params": start_params,
        **kwargs,
    }
    # Only pass solver to GeneralizedLinearRegressor (custom classes don't need it)
    if regressor_class is GeneralizedLinearRegressor:
        glm_arguments["solver"] = solver
        glm_arguments.pop("n_c_updates", None)

    if fit_theta_iter:
        assert family_is_negative_binomial(
            kwargs["family"]
        ), "Iterative theta fitting only supported for Negative Binomial family."
        assert not regressorCV, "regressorCV not supported for IterativeThetaGLM"

        theta_init = _compute_theta_init(
            h_start=h_start,
            w_start=w_start,
            c_start=c_start,
            theta_cal=theta_cal,
            df=df,
        )

        logger.debug("Calculating theta_init = %s", theta_init)

        regressor = _build_iterative_regressor(
            sklearnCV=sklearnCV,
            fit_theta_iter_loc=fit_theta_iter_loc,
            theta_init=theta_init,
            fit_theta_max_iter=fit_theta_max_iter,
            glm_arguments=glm_arguments,
            cv=cv,
            warm_start_alpha=warm_start_alpha,
            parallel=parallel,
            regressorCV_one_SE_rule=sklearnCV_one_SE_rule,
            regressor_class=regressor_class,
            reestimate_c_on_val=reestimate_c_on_val,
        )

    else:
        regressor = _build_non_iterative_regressor(
            regressorCV=regressorCV,
            sklearnCV=sklearnCV,
            cv=cv,
            warm_start_alpha=warm_start_alpha,
            parallel=parallel,
            glm_arguments=glm_arguments,
            sklearnCV_one_SE_rule=sklearnCV_one_SE_rule,
            regressor_class=regressor_class,
            reestimate_c_on_val=reestimate_c_on_val,
        )

    regressor.fit(X, y, offset=offset)

    # check that our procedure to select the start coef is correct. If wrong, probably the orders of categories logic was wrongly inferred.

    if not (start_params is None):
        assert regressor.feature_names_ == start_features_names

    if freeze_c:
        h_hat, w_hat, c_hat = glum_coef_to_hwc_frozen_c(
            regressor.coef_,
            regressor.intercept_,
            regressor.feature_names_,
            c_start,
            dropped_levels_dict,
        )
    else:
        h_hat, w_hat, c_hat = glum_coef_to_hwc(
            regressor.coef_,
            regressor.intercept_,
            regressor.feature_names_,
            dropped_levels_dict,
        )

    return h_hat, w_hat, c_hat, regressor


def _standardize_and_prepare(X_tabmat, P2_mask, sample_weight, alpha, l1_ratio,
                              start_params):
    """Standardize X and P2 to match glum's preprocessing.

    Returns (X_std, col_means, col_stds, P2, coef, sample_weight).
    sample_weight is normalized to sum to 1 (matching glum).
    """
    from glum._glm import _standardize, _standardize_warm_start

    # Normalize sample_weight to sum to 1 (glum._glm.py:2567-2568)
    weights_sum = np.sum(sample_weight)
    sample_weight = sample_weight / weights_sum

    n_features = X_tabmat.shape[1]
    P2_no_alpha = (1 - l1_ratio) * np.array(P2_mask, dtype=np.float64)
    P1_dummy = np.zeros(n_features, dtype=np.float64)

    X_std, col_means, col_stds, _, _, _, _, P2_no_alpha = _standardize(
        X_tabmat, sample_weight,
        center_predictors=True,             # = fit_intercept
        estimate_as_if_scaled_model=False,   # = scale_predictors default
        lower_bounds=None, upper_bounds=None, A_ineq=None,
        P1=P1_dummy,
        P2=P2_no_alpha,
    )
    P2 = alpha * P2_no_alpha

    n_coef = n_features + 1
    if start_params is not None:
        coef = np.array(start_params, dtype=np.float64)
        _standardize_warm_start(coef, col_means, col_stds)
    else:
        coef = np.zeros(n_coef, dtype=np.float64)

    return X_std, col_means, col_stds, P2, coef, sample_weight


def _unstandardize_coef(coef_out, col_means, col_stds):
    """Unstandardize intercept and coefficients (matching glum)."""
    from glum._glm import _unstandardize
    intercept = float(coef_out[0])
    coef_features = coef_out[1:].copy()
    intercept, coef_features = _unstandardize(
        col_means, col_stds, intercept, coef_features,
    )
    return intercept, coef_features


def _fit_coordinate_descent(df, h_start, w_start, c_start, **kwargs):
    """Coordinate descent path: freeze_c=True LBFGS on h,w + analytical c update.

    Uses the same tabmat / glum infrastructure as the freeze_c=True glum path,
    but replaces glum's _lbfgs_solver with coordinate_descent_lbfgs_solver
    which inserts a c update after every LBFGS iteration.
    """
    from glum._glm import get_family, get_link

    # ---- categorise (i, j) only, like freeze_c=True ---------------------
    categoricals, glm_categorizer, categories_dict = categorizer_glum(df, freeze_c=True)
    X_df = glm_categorizer.fit_transform(df[categoricals])
    y = df["k"].values.astype(np.float64)

    dropped_levels_dict = {key: cats[0] for key, cats in categories_dict.items()}

    # ---- build tabmat SplitMatrix once -----------------------------------
    X_tabmat = tm.from_pandas(
        X_df, drop_first=True,
        categorical_format="{name}[{category}]",
    )

    # ---- c initialisation ------------------------------------------------
    if c_start is None:
        raise ValueError("c_start required for coordinate_descent solver.")
    offset = c_to_offset(c_start, df["p"]).astype(np.float64)

    # keep a per-nucleus log_c array in sync with offset
    _, _, cats_p_dict = categorizer_glum(df, freeze_c=False)
    levels_p = cats_p_dict["p"]
    p_level_to_idx = {level: idx for idx, level in enumerate(levels_p)}
    p_idx = df["p"].map(p_level_to_idx).values.astype(np.intp)
    n_p = len(levels_p)
    log_c = np.log(
        c_start.reindex(levels_p).fillna(1.0).clip(lower=1e-12).values
    ).astype(np.float64)

    # ---- h, w warm-start (same format as freeze_c=True) ------------------
    if h_start is not None and w_start is not None:
        start_params, _, _ = h_w_to_glum_coef(
            h_start, w_start,
            levels_i=categories_dict["i"],
            levels_j=categories_dict["j"],
        )
    else:
        start_params = None

    # ---- prepare GLM internals ------------------------------------------
    family = get_family(kwargs["family"])
    link = get_link("log", family)
    theta = getattr(family, "theta", None)

    alpha = kwargs.get("alpha", 1.0)
    l1_ratio = kwargs.get("l1_ratio", 0)
    from src.destriping.GLUM.penalties import P_hw_only_from_df
    P2_mask = P_hw_only_from_df(df, freeze_c=True)

    sample_weight = np.ones(len(y), dtype=np.float64)
    max_iter = kwargs.get("max_iter", 100_000)

    # ---- standardize (matching glum) ------------------------------------
    X_std, col_means, col_stds, P2, coef, sample_weight = _standardize_and_prepare(
        X_tabmat, P2_mask, sample_weight, alpha, l1_ratio, start_params,
    )

    # ---- run coordinate descent solver -----------------------------------
    coef_out, n_iter, _, _ = coordinate_descent_lbfgs_solver(
        coef=coef,
        X=X_std,
        y=y,
        sample_weight=sample_weight,
        P2=P2,
        verbose=False,
        family=family,
        link=link,
        max_iter=max_iter,
        tol=1e-4,
        offset=offset,
        p_idx=p_idx,
        n_p=n_p,
        log_c=log_c,
        theta=theta,
    )

    # ---- unstandardize and build result ---------------------------------
    intercept, coef_hw = _unstandardize_coef(coef_out, col_means, col_stds)

    # Recover fitted c from log_c (which was updated in-place by the solver)
    fitted_c = pd.Series(np.exp(log_c), index=levels_p)

    # feature_names in glum's format
    feature_names = []
    for col in categoricals:
        for level in categories_dict[col][1:]:  # skip dropped first level
            feature_names.append(f"{col}[{level}]")

    regressor = SimpleNamespace(
        coef_=coef_hw,
        intercept_=intercept,
        feature_names_=feature_names,
        n_iter_=n_iter,
        max_iter=max_iter,
        family=family,
    )

    h_hat, w_hat, c_hat = glum_coef_to_hwc_frozen_c(
        coef_hw, intercept, feature_names,
        fitted_c, dropped_levels_dict,
    )

    return h_hat, w_hat, c_hat, regressor


def _fit_lbfgs_control(df, h_start, w_start, c_start, freeze_c, **kwargs):
    """Control: plain LBFGS via setulb on all variables (p, i, j).

    Same loop as coordinate_descent but without profiling out c.
    Used for apples-to-apples timing comparison.
    """
    from glum._glm import get_family, get_link

    categoricals, glm_categorizer, categories_dict = categorizer_glum(df, freeze_c)
    X_df = glm_categorizer.fit_transform(df[categoricals])
    y = df["k"].values.astype(np.float64)

    dropped_levels_dict = {key: cats[0] for key, cats in categories_dict.items()}

    X_tabmat = tm.from_pandas(
        X_df, drop_first=True,
        categorical_format="{name}[{category}]",
    )

    # offset (same logic as original glum path)
    if freeze_c:
        if c_start is None:
            raise ValueError("c_start must be provided when freeze_c = True.")
        offset = c_to_offset(c_start, df["p"]).astype(np.float64)
    else:
        offset = None

    # start_params
    if freeze_c:
        if h_start is not None and w_start is not None:
            start_params, _, _ = h_w_to_glum_coef(
                h_start, w_start,
                levels_i=categories_dict["i"],
                levels_j=categories_dict["j"],
            )
        else:
            start_params = None
    else:
        if h_start is not None and w_start is not None and c_start is not None:
            start_params, _, _ = h_w_c_to_glum_coef(
                h_start, w_start, c_start,
                levels_i=categories_dict["i"],
                levels_j=categories_dict["j"],
                levels_p=categories_dict["p"],
            )
        else:
            start_params = None

    family = get_family(kwargs["family"])
    link = get_link("log", family)

    alpha = kwargs.get("alpha", 1.0)
    l1_ratio = kwargs.get("l1_ratio", 0)
    P2_mask = kwargs.get("P2")
    if P2_mask is None:
        from src.destriping.GLUM.penalties import P_hw_only_from_df
        P2_mask = P_hw_only_from_df(df, freeze_c=freeze_c)

    sample_weight = np.ones(len(y), dtype=np.float64)
    max_iter = kwargs.get("max_iter", 100_000)

    # ---- standardize (matching glum) ------------------------------------
    X_std, col_means, col_stds, P2, coef, sample_weight = _standardize_and_prepare(
        X_tabmat, P2_mask, sample_weight, alpha, l1_ratio, start_params,
    )

    coef_out, n_iter, _, _ = lbfgs_control_solver(
        coef=coef,
        X=X_std,
        y=y,
        sample_weight=sample_weight,
        P2=P2,
        verbose=False,
        family=family,
        link=link,
        max_iter=max_iter,
        tol=1e-4,
        offset=offset,
    )

    # ---- unstandardize and build result ---------------------------------
    intercept, coef_features = _unstandardize_coef(coef_out, col_means, col_stds)

    feature_names = []
    for col in categoricals:
        for level in categories_dict[col][1:]:
            feature_names.append(f"{col}[{level}]")

    regressor = SimpleNamespace(
        coef_=coef_features,
        intercept_=intercept,
        feature_names_=feature_names,
        n_iter_=n_iter,
        max_iter=max_iter,
        family=family,
    )

    if freeze_c:
        h_hat, w_hat, c_hat = glum_coef_to_hwc_frozen_c(
            coef_features, intercept, feature_names,
            c_start, dropped_levels_dict,
        )
    else:
        h_hat, w_hat, c_hat = glum_coef_to_hwc(
            coef_features, intercept, feature_names,
            dropped_levels_dict,
        )

    return h_hat, w_hat, c_hat, regressor
