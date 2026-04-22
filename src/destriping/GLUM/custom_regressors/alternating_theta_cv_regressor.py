"""Alternating theta-CV regressor.

Alternates between:
  1. Full alpha CV at current theta  →  best_alpha
  2. Theta iteration to convergence at fixed best_alpha  (cheap: 1 fit per step)
  3. Re-run alpha CV only if theta changed

Stops when theta stabilises across an outer iteration, meaning the
alpha selected by CV is already consistent with the converged theta.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from glum._glm import ArrayLike

from src.destriping.GLUM.custom_regressors.cv_regressor import CustomCVRegressor
from src.destriping.GLUM.custom_regressors.helpers import (
    RegressorTypeLike,
    delegate_getattr,
    extract_intercept_coef,
    remove_prefix_from_kwargs,
    wrap_dict_with_prefix,
)
from src.destriping.GLUM.custom_regressors.warm_start_wrapper import WarmStartWrapper
from src.destriping.GLUM.glum_nb_helpers import set_family_arg
from src.destriping.GLUM.iterative_theta import theta_md

logger = logging.getLogger("Alternating Theta CV")


class AlternatingThetaCVRegressor:
    """Alternating block-coordinate optimisation of (alpha, theta).

    Outer loop:
        1.  Run full alpha CV at current theta  →  best_alpha
        2.  Iterate theta to convergence at fixed best_alpha
        3.  If theta did not change  →  stop; otherwise go to 1.

    The inner theta loop is cheap (one model fit per step at a single
    alpha) compared to the full CV (n_alphas × n_folds fits).  The outer
    loop typically converges in 1–2 passes.
    """

    def __init__(
        self,
        param_grid,
        one_SE_rule,
        cv,
        warm_start_alpha,
        parallel,
        regressor_class: RegressorTypeLike,
        delta_theta_thresh=1e-3,
        theta_max_iter=3,
        theta_init=1.0,
        max_outer_iter=5,
        reestimate_c_on_val: bool = False,
        **regressor_args,
    ):
        self.param_grid = param_grid
        self.one_SE_rule = one_SE_rule
        self.cv = cv
        self.warm_start_alpha = warm_start_alpha
        self.parallel = parallel
        self.regressor_class = regressor_class
        self.delta_theta_thresh = delta_theta_thresh
        self.theta_max_iter = theta_max_iter
        self.theta_init = theta_init
        self.max_outer_iter = max_outer_iter
        self.reestimate_c_on_val = reestimate_c_on_val

        sub_params = remove_prefix_from_kwargs(regressor_args, "regressor")
        self.regressor_args = dict(sub_params)

        # Set after fit()
        self.regressor = None
        self.outer_iter_ = None
        self.theta_history_ = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _theta_cal(self, X, y, offset, glm):
        """Estimate theta from a fitted model (method-of-moments)."""
        dfr = len(y) - len(glm.coef_)
        mu = glm.predict(X, offset=offset)
        return theta_md(y, mu, dfr)

    def _build_cv_regressor(self, theta):
        """Build a fresh CustomCVRegressor at the given theta."""
        args = {**self.regressor_args, "family": set_family_arg(theta)}
        return CustomCVRegressor(
            param_grid=self.param_grid,
            one_SE_rule=self.one_SE_rule,
            regressor_class=self.regressor_class,
            cv=self.cv,
            warm_start_alpha=self.warm_start_alpha,
            parallel=self.parallel,
            reestimate_c_on_val=self.reestimate_c_on_val,
            **wrap_dict_with_prefix(args, "regressor"),
        )

    def _build_single_regressor(self, theta, best_glm_params, warm_start=None):
        """Build a regressor for a single fit at fixed alpha (theta iteration).

        If warm_start_alpha is enabled, wraps in WarmStartWrapper to
        traverse the alpha path.  Otherwise builds a plain regressor.
        """
        args = {**self.regressor_args, **best_glm_params, "family": set_family_arg(theta)}

        if warm_start is not None:
            args["start_params"] = warm_start

        if self.warm_start_alpha and "_alpha_path" in best_glm_params:
            alpha_path = args.pop("_alpha_path")
            return WarmStartWrapper(
                regressor_class=self.regressor_class,
                alpha_param_key=self._cv_regressor.best_estimator.alpha_param_key,
                start_params_key=self._cv_regressor.best_estimator.start_params_key,
                alpha_path=alpha_path,
                **wrap_dict_with_prefix(args, "regressor"),
            )
        else:
            return self.regressor_class(**args)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ):
        theta = self.theta_init
        self.theta_history_ = [theta]
        fit_kw = dict(sample_weight=sample_weight, offset=offset)

        for outer_iter in range(1, self.max_outer_iter + 1):
            self.outer_iter_ = outer_iter

            # ---- Step 1: full alpha CV at current theta ------------------
            logger.debug(
                f"Outer iter {outer_iter}: alpha CV with theta={theta:.6f}"
            )
            self._cv_regressor = self._build_cv_regressor(theta)
            self._cv_regressor.fit(X=X, y=y, **fit_kw)

            best_glm_params = self._cv_regressor.best_glm_params
            logger.debug(
                f"Outer iter {outer_iter}: best_alpha={best_glm_params.get('alpha', '?')}"
            )

            # Warm-start the theta loop from the CV's best estimator
            warm_start = extract_intercept_coef(self._cv_regressor.best_estimator)

            # ---- Step 2: iterate theta to convergence at fixed alpha -----
            prev_theta = theta
            for theta_step in range(1, self.theta_max_iter + 1):
                reg = self._build_single_regressor(theta, best_glm_params, warm_start)
                reg.fit(X, y, **fit_kw)

                new_theta = self._theta_cal(X, y, offset, reg)
                logger.debug(
                    f"  Theta step {theta_step}: theta {theta:.6f} → {new_theta:.6f}"
                )
                self.theta_history_.append(new_theta)

                # Warm-start next theta step from this fit
                warm_start = extract_intercept_coef(reg)

                if abs(new_theta - theta) < self.delta_theta_thresh:
                    logger.debug(f"  Theta converged (delta={abs(new_theta - theta):.2e})")
                    theta = new_theta
                    break
                theta = new_theta
            else:
                logger.debug(f"  Theta max_iter ({self.theta_max_iter}) reached")

            # Store the last fitted regressor from the theta loop
            self.regressor = reg

            # ---- Step 3: check if theta changed --------------------------
            if abs(theta - prev_theta) < self.delta_theta_thresh:
                logger.debug(
                    f"Outer loop converged: theta stable at {theta:.6f} "
                    f"after {outer_iter} outer iteration(s)"
                )
                break
        else:
            logger.debug(
                f"Outer loop: max_outer_iter ({self.max_outer_iter}) reached"
            )

        return self

    __getattr__ = delegate_getattr("regressor", "regressor_class")
