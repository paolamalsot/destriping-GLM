"""sklearn-compatible regressor classes for coordinate descent and LBFGS control.

These plug into the existing CV / theta-iteration wrapper chain
(IterativeThetaGLM, CustomCVRegressor) as drop-in replacements for
GeneralizedLinearRegressor.
"""

import numpy as np
import pandas as pd
import tabmat as tm

from src.destriping.GLUM.coordinate_descent_solver import (
    coordinate_descent_lbfgs_solver,
)
from src.destriping.GLUM.fit import (
    _standardize_and_prepare,
    _unstandardize_coef,
    extract_categories_dict_from_categorizer,
    factor_to_glum_coef,
    glum_coef_to_hwc,
    h_w_to_glum_coef,
    c_to_offset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _categories_from_X(X):
    """Extract category lists from a categorized DataFrame."""
    cats = {}
    for col in X.columns:
        if hasattr(X[col], "cat"):
            cats[col] = X[col].cat.categories.tolist()
    return cats


def _feature_names_from_categories(categories_dict, categoricals):
    """Build glum-style feature names: '{col}[{level}]', dropping first level."""
    names = []
    for col in categoricals:
        for level in categories_dict[col][1:]:
            names.append(f"{col}[{level}]")
    return names


def _coef_to_hwc(coef, intercept, feature_names, categories_dict):
    """Convert full (p,i,j) glum coef back to h, w, c Series."""
    dropped_levels_dict = {key: cats[0] for key, cats in categories_dict.items()}
    return glum_coef_to_hwc(coef, intercept, feature_names, dropped_levels_dict)


def _build_full_coef(intercept_hw, coef_hw, log_c, categories_dict, feature_names_ij):
    """Reconstruct full (p,i,j) glum coef from h,w coef and fitted log_c.

    Returns (intercept, coef, feature_names) in glum format.
    """
    levels_p = categories_dict["p"]
    dropped_p = levels_p[0]

    # p contribution: relative to dropped level
    intercept_p = log_c[0]  # dropped level's value
    coef_p = log_c[1:] - intercept_p

    intercept = float(intercept_hw) + intercept_p
    coef = np.concatenate([coef_p, coef_hw])

    # feature names: p names + i,j names
    p_names = [f"p[{level}]" for level in levels_p[1:]]
    feature_names = p_names + feature_names_ij

    return intercept, coef, feature_names


# ---------------------------------------------------------------------------
# Base class with shared init / get_params / set_params
# ---------------------------------------------------------------------------


class _BaseCustomRegressor:
    """Base for CoordinateDescentRegressor and LBFGSControlRegressor."""

    _init_params = [
        "family",
        "alpha",
        "P2",
        "l1_ratio",
        "link",
        "fit_intercept",
        "drop_first",
        "start_params",
        "max_iter",
        "n_c_updates",
    ]

    def __init__(
        self,
        family=None,
        alpha=1.0,
        P2="identity",
        l1_ratio=0,
        link="log",
        fit_intercept=True,
        drop_first=True,
        start_params=None,
        max_iter=100_000,
        n_c_updates=1,
        **kwargs,
    ):
        self.family = family
        self.alpha = alpha
        self.P2 = P2
        self.l1_ratio = l1_ratio
        self.link = link
        self.fit_intercept = fit_intercept
        self.drop_first = drop_first
        self.start_params = start_params
        self.max_iter = max_iter
        self.n_c_updates = n_c_updates
        # store extra kwargs for forward-compatibility
        for k, v in kwargs.items():
            setattr(self, k, v)
        self._extra_params = list(kwargs.keys())

    def get_params(self, deep=True):
        params = {name: getattr(self, name) for name in self._init_params}
        for name in self._extra_params:
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
            if key not in self._init_params and key not in self._extra_params:
                self._extra_params.append(key)
        return self

    @property
    def _family_instance(self):
        from glum._glm import get_family

        return get_family(self.family) if isinstance(self.family, str) else self.family

    _ETA_CLIP = 30.0

    def predict(self, X, offset=None):
        """Predict mu = exp(clip(X @ coef + intercept + offset))."""
        categories_dict = _categories_from_X(X)
        categoricals = [c for c in ["p", "i", "j"] if c in categories_dict]
        X_tabmat = tm.from_pandas(
            X[categoricals],
            drop_first=True,
            categorical_format="{name}[{category}]",
        )
        eta = self.intercept_ + X_tabmat @ self.coef_
        if offset is not None:
            eta = eta + np.asarray(offset)
        np.clip(eta, -self._ETA_CLIP, self._ETA_CLIP, out=eta)
        return np.exp(eta)


# ---------------------------------------------------------------------------
# CoordinateDescentRegressor
# ---------------------------------------------------------------------------


class CoordinateDescentRegressor(_BaseCustomRegressor):
    """LBFGS on (h, w) with analytical c update — sklearn-compatible interface.

    Accepts X with (p, i, j) categorical columns. Internally splits into
    (i, j) for the LBFGS solver and handles c via bincount updates.
    """

    def fit(self, X, y, sample_weight=None, offset=None):
        from glum._glm import get_family, get_link

        y = np.asarray(y, dtype=np.float64)
        categories_dict = _categories_from_X(X)
        categoricals_pij = [c for c in ["p", "i", "j"] if c in categories_dict]

        # ---- split: (i,j) for LBFGS, p for c update -----------------------
        assert (
            "p" in categories_dict
        ), "X must contain a 'p' column for coordinate descent."
        levels_p = categories_dict["p"]
        levels_i = categories_dict["i"]
        levels_j = categories_dict["j"]

        # p index mapping
        p_level_to_idx = {level: idx for idx, level in enumerate(levels_p)}
        p_idx = X["p"].map(p_level_to_idx).values.astype(np.intp)
        n_p = len(levels_p)

        # tabmat CategoricalMatrix for fast group-sum / broadcast in c-update
        import pandas as pd

        p_cat_pd = pd.Categorical.from_codes(p_idx, categories=np.arange(n_p))
        P_cat = tm.CategoricalMatrix(p_cat_pd, drop_first=False)

        # tabmat from (i, j) only
        X_ij = X[["i", "j"]]
        X_tabmat = tm.from_pandas(
            X_ij,
            drop_first=True,
            categorical_format="{name}[{category}]",
        )

        # ---- family / link --------------------------------------------------
        family = (
            get_family(self.family) if isinstance(self.family, str) else self.family
        )
        link = get_link("log", family)
        theta = getattr(family, "theta", None)

        # ---- derive h_start, w_start, c_start from start_params ------------
        if self.start_params is not None:
            feature_names_pij = _feature_names_from_categories(
                categories_dict,
                categoricals_pij,
            )
            dropped_pij = {k: cats[0] for k, cats in categories_dict.items()}
            h_start, w_start, c_start = _coef_to_hwc(
                self.start_params[1:],
                self.start_params[0],
                feature_names_pij,
                categories_dict,
            )
        else:
            h_start, w_start, c_start = None, None, None

        # ---- c initialization -----------------------------------------------
        if c_start is not None:
            log_c = np.log(
                c_start.reindex(levels_p).fillna(1.0).clip(lower=1e-12).values
            ).astype(np.float64)
            offset_arr = log_c[p_idx].copy()
        else:
            log_c = np.zeros(n_p, dtype=np.float64)
            offset_arr = np.zeros(len(y), dtype=np.float64)

        # ---- h, w start params for (i, j) LBFGS ----------------------------
        if h_start is not None and w_start is not None:
            hw_start, _, _ = h_w_to_glum_coef(
                h_start,
                w_start,
                levels_i=levels_i,
                levels_j=levels_j,
            )
        else:
            hw_start = None

        # ---- P2 for (i, j) and c --------------------------------------------
        n_i = len(levels_i) - 1  # after drop_first
        n_j = len(levels_j) - 1
        n_p_coefs = len(levels_p) - 1
        if isinstance(self.P2, np.ndarray):
            # P2 is for (p,i,j) — extract parts
            P2_c_mask = np.concatenate(
                [[0.0], self.P2[:n_p_coefs]]
            )  # len n_p, 0 for dropped
            P2_mask = self.P2[n_p_coefs:]
        elif self.P2 == "identity":
            P2_c_mask = np.ones(n_p, dtype=np.float64)
            P2_mask = np.ones(n_i + n_j, dtype=np.float64)
        else:  # "hw_only"
            P2_c_mask = np.zeros(n_p, dtype=np.float64)
            P2_mask = np.ones(n_i + n_j, dtype=np.float64)

        # ---- sample weight ---------------------------------------------------
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        assert np.allclose(
            sample_weight, 1.0
        ), "Coordinate descent c penalty assumes uniform sample weights."
        assert (
            self.l1_ratio == 0
        ), "L1 penalty not supported in coordinate descent (l1_ratio must be 0)."

        # ---- standardize and prepare ----------------------------------------
        X_std, col_means, col_stds, P2, coef, sample_weight = _standardize_and_prepare(
            X_tabmat,
            P2_mask,
            sample_weight,
            self.alpha,
            self.l1_ratio,
            hw_start,
        )

        # ---- P2_c scaling (no column standardization, uniform weights) ------
        n_obs = len(y)
        P2_c = self.alpha * n_obs * P2_c_mask

        # ---- run solver ------------------------------------------------------
        coef_out, n_iter, _, _ = coordinate_descent_lbfgs_solver(
            coef=coef,
            X=X_std,
            y=y,
            sample_weight=sample_weight,
            P2=P2,
            verbose=False,
            family=family,
            link=link,
            max_iter=self.max_iter,
            tol=1e-4,
            offset=offset_arr,
            p_idx=p_idx,
            n_p=n_p,
            log_c=log_c,
            P_cat=P_cat,
            theta=theta,
            n_c_updates=self.n_c_updates,
            P2_c=P2_c,
        )

        # ---- unstandardize ---------------------------------------------------
        intercept_hw, coef_hw = _unstandardize_coef(coef_out, col_means, col_stds)

        # ---- build feature names for (i,j) ----------------------------------
        feature_names_ij = _feature_names_from_categories(
            {"i": levels_i, "j": levels_j},
            ["i", "j"],
        )

        # ---- reconstruct full (p,i,j) coef ----------------------------------
        self.intercept_, self.coef_, self.feature_names_ = _build_full_coef(
            intercept_hw,
            coef_hw,
            log_c,
            categories_dict,
            feature_names_ij,
        )
        self.n_iter_ = n_iter
        self.family = family
        return self


# ---------------------------------------------------------------------------
# LBFGSControlRegressor
# ---------------------------------------------------------------------------


class LBFGSControlRegressor(_BaseCustomRegressor):
    """Plain LBFGS on all variables (p, i, j) — sklearn-compatible interface.

    Same as GeneralizedLinearRegressor with solver='lbfgs' but using our
    setulb loop. For apples-to-apples comparison with CoordinateDescentRegressor.
    """

    def fit(self, X, y, sample_weight=None, offset=None):
        from glum._glm import get_family, get_link

        y = np.asarray(y, dtype=np.float64)
        categories_dict = _categories_from_X(X)
        categoricals = [c for c in ["p", "i", "j"] if c in categories_dict]

        # ---- build tabmat from all columns -----------------------------------
        X_tabmat = tm.from_pandas(
            X[categoricals],
            drop_first=True,
            categorical_format="{name}[{category}]",
        )

        # ---- family / link ---------------------------------------------------
        family = (
            get_family(self.family) if isinstance(self.family, str) else self.family
        )
        link = get_link("log", family)

        # ---- P2 --------------------------------------------------------------
        n_features = X_tabmat.shape[1]
        if isinstance(self.P2, np.ndarray):
            P2_mask = self.P2
        elif self.P2 == "identity":
            P2_mask = np.ones(n_features, dtype=np.float64)
        else:
            P2_mask = np.ones(n_features, dtype=np.float64)

        # ---- sample weight ---------------------------------------------------
        if sample_weight is None:
            sample_weight = np.ones(len(y), dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # ---- offset ----------------------------------------------------------
        if offset is not None:
            offset = np.asarray(offset, dtype=np.float64)

        # ---- standardize and prepare ----------------------------------------
        X_std, col_means, col_stds, P2, coef, sample_weight = _standardize_and_prepare(
            X_tabmat,
            P2_mask,
            sample_weight,
            self.alpha,
            self.l1_ratio,
            self.start_params,
        )

        # ---- run solver ------------------------------------------------------
        coef_out, n_iter, _, _ = lbfgs_control_solver(
            coef=coef,
            X=X_std,
            y=y,
            sample_weight=sample_weight,
            P2=P2,
            verbose=False,
            family=family,
            link=link,
            max_iter=self.max_iter,
            tol=1e-4,
            offset=offset,
        )

        # ---- unstandardize ---------------------------------------------------
        self.intercept_, self.coef_ = _unstandardize_coef(coef_out, col_means, col_stds)

        # ---- feature names ---------------------------------------------------
        self.feature_names_ = _feature_names_from_categories(
            categories_dict, categoricals
        )
        self.n_iter_ = n_iter
        self.family = family
        return self
