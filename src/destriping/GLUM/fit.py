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


def extract_coef(coef, feature_names):
    # coef is a numpy array with same size as feature_names
    # feature_names are glm.feature_names with the categorical-format '{name}[{category}]'
    # returns a pandas df with columns name, column category and corresponding value
    assert len(coef) == len(
        feature_names
    )  # make sure that there is no feature_name for the intercept ?
    df = pd.DataFrame(data={"feature_names": feature_names, "coef": coef})
    df[["name", "category"]] = df["feature_names"].str.extract(r"^(.*?)\[(.*?)\]$")
    return df


def extract_coef_specific(coef, feature_names, var_name):
    coef_df = extract_coef(coef, feature_names)
    coef_specific = (
        coef_df.query("name == @var_name").copy().set_index(keys=["category"])["coef"]
    )
    return coef_specific


def rescale_hwc(h: pd.Series, w: pd.Series, c: pd.Series, exp_intercept: float):
    # Rescale the solutions h, w, c, exp_intercept such that we get an equivalent model mu[ijp] = h_s[i] * w_s[j] * c_s[p]
    # with sum(h_s) = len(h_s) and sum(w_s) = len(w_s)
    f_h = len(h) / np.sum(h)
    f_w = len(w) / np.sum(w)
    f_c = exp_intercept / (f_h * f_w)
    return h * f_h, w * f_w, c * f_c


def extract_coef_with_dropped_level(
    glum_coef, glum_feature_names, coef_key, dropped_level
):
    coef_specific = extract_coef_specific(glum_coef, glum_feature_names, coef_key)
    assert not (dropped_level in coef_specific.index)
    coef_specific.loc[dropped_level] = 0.0
    coef_specific = coef_specific.apply(np.exp)
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

    h, w, c = rescale_hwc(**coef_dict, exp_intercept=np.exp(intercept))
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
                **wrap_dict_with_prefix(inner_regressor_args, "regressor"),
            )

        case "out":
            inner_regressor_args = {
                "param_grid": param_grid,
                "one_SE_rule": regressorCV_one_SE_rule,
                "regressor_class": GeneralizedLinearRegressor,
                "cv": cv,
                "warm_start_alpha": warm_start_alpha,
                "parallel": parallel,
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
                regressor_class=GeneralizedLinearRegressor,
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
        )

    # fit_theta_iter and no sklearnCV
    return IterativeThetaGLM(
        theta_max_iter=fit_theta_max_iter,
        theta_init=theta_init,
        regressor_class=GeneralizedLinearRegressor,
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
            regressor_class=GeneralizedLinearRegressor,
            cv=cv,
            parallel=parallel,
            warm_start_alpha=warm_start_alpha,
            **wrap_dict_with_prefix(glm_arguments, "regressor"),
        )

    return GeneralizedLinearRegressor(**glm_arguments)


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
        "l1_ratio": 0,
        "fit_intercept": True,
        "drop_first": True,
        "start_params": start_params,
        **kwargs,
    }

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
