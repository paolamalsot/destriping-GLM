from __future__ import annotations
from src.destriping.GLUM.custom_regressors.helpers import remove_prefix_from_kwargs

from src.destriping.GLUM.custom_regressors.helpers import remove_prefix_from_kwargs

import logging

from glum import GeneralizedLinearRegressor


from src.destriping.GLUM.custom_regressors.helpers import RegressorTypeLike
from src.destriping.GLUM.glum_nb_helpers import set_family_arg

from glum import GeneralizedLinearRegressor

from glum._glm import ArrayLike
from src.destriping.GLUM.iterative_theta import theta_md

from glum import GeneralizedLinearRegressor
from typing import Optional
from src.destriping.GLUM.custom_regressors.helpers import delegate_getattr

import logging

logger = logging.getLogger("Iterative Theta GLM")

fit_theta_iter_status_keys = ["theta_iter_", "theta_iter_converged_"]


class IterativeThetaGLM:
    """
    Wrapper around Regressor with an outer iterative loop on theta.

    - At each iteration, we create a *fresh* Regressor(**fitting_kwargs) with updated theta
      and fit it.
    - The final fitted model is stored in self.regressor
    - Any attribute / method not explicitely defined is delegated to self.regressor via __getattr__.
    """

    def __init__(
        self,
        delta_theta_thresh=10 ** (-3),
        theta_max_iter=3,
        regressor_class: RegressorTypeLike = GeneralizedLinearRegressor,
        theta_init=1.0,
        family_arg_name="family",  # could be regressor__family if nested..
        **regressor_args,
    ):
        # NB regressor_args must be named regressor__{}
        self.delta_theta_thresh = delta_theta_thresh
        self.theta_max_iter = theta_max_iter
        self.regressor_class = regressor_class
        self.theta_init = theta_init

        if regressor_args is None:
            regressor_args = {}
        sub_params = remove_prefix_from_kwargs(regressor_args, "regressor")
        self.regressor_args = dict(sub_params)  # just for pylance
        self.regressor = None
        self.family_arg_name = family_arg_name
        self.theta_iter_converged_ = False
        self.theta_iter_ = None

    def theta_cal(self, X, y, offset, glm):
        dfr = len(y) - len(glm.coef_)
        mu = glm.predict(X, offset=offset)
        theta = theta_md(y, mu, dfr)
        return theta

    def update_regressor_args(self, theta):
        return {**self.regressor_args, self.family_arg_name: set_family_arg(theta)}

    def fit_regressor(self, X, y, theta, sample_weight=None, offset=None):
        regressor_args = self.update_regressor_args(theta=theta)
        self.regressor = self.regressor_class(**regressor_args)
        self.regressor.fit(X, y, sample_weight=sample_weight, offset=offset)

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ):
        theta = self.theta_init
        logger.debug(f"IterativeThetaGLM: setting initial theta to {self.theta_init}")
        self.theta_iter_ = 1
        while True:
            logger.debug(f"IterativeThetaGLM: {self.theta_iter_=}")
            self.fit_regressor(
                X, y=y, theta=theta, sample_weight=sample_weight, offset=offset
            )
            new_theta = self.theta_cal(X, y, offset, self.regressor)
            logger.debug(f"IterativeThetaGLM: new_theta = {new_theta}")
            diff_theta = abs(new_theta - theta)
            self.theta_iter_converged_ = diff_theta < self.delta_theta_thresh
            if self.theta_iter_converged_:
                logger.debug(
                    f"IterativeThetaGLM: diff_delta ({diff_theta:.3e})< delta_thresh ({self.delta_theta_thresh:.3e})"
                )
                break
            if self.theta_iter_ >= self.theta_max_iter:
                logger.debug("IterativeThetaGLM: max iter reached")
                break
            self.theta_iter_ += 1
            theta = new_theta
        theta = new_theta
        return self

    __getattr__ = delegate_getattr("regressor", "regressor_class")
