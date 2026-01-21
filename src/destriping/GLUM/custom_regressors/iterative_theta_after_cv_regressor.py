from __future__ import annotations
from src.destriping.GLUM.custom_regressors.cv_regressor import CustomCVRegressor
from src.destriping.GLUM.custom_regressors.helpers import remove_prefix_from_kwargs, wrap_dict_with_prefix
from src.destriping.GLUM.custom_regressors.iterative_theta_regressor import IterativeThetaGLM
import logging
from typing import TypeAlias
from typing import Type, TypeAlias
from glum import GeneralizedLinearRegressor
from glum._glm import GeneralizedLinearRegressorBase
from typing import Any, NamedTuple, Optional, Union, cast
from src.destriping.GLUM.custom_regressors.cv_regressor import CustomCVRegressor
from src.destriping.GLUM.custom_regressors.iterative_theta_regressor import (
    IterativeThetaGLM,
)
from src.destriping.GLUM.cv import _start_params_key
from src.destriping.GLUM.custom_regressors.helpers import RegressorTypeLike
from src.destriping.GLUM.glum_nb_helpers import set_family_arg
from src.destriping.GLUM.custom_regressors.helpers import extract_intercept_coef
from typing import TypeAlias
from typing import Type, TypeAlias
from glum import GeneralizedLinearRegressor
from glum._glm import GeneralizedLinearRegressorBase
from typing import NamedTuple, Optional, Union, cast
from glum._glm import ArrayLike
from src.destriping.GLUM.custom_regressors.warm_start_wrapper import WarmStartWrapper

logger = logging.getLogger("Iterative Theta After CV")

class IterativeThetaAfterCVRegressor():

    def __init__(
        self,
        param_grid,
        one_SE_rule,
        cv,
        warm_start_alpha,
        parallel,
        regressor_class: RegressorTypeLike,
        delta_theta_thresh=10 ** (-3),
        theta_max_iter=3,
        theta_init=1.0,
        **regressor_args
    ):
        self.param_grid = param_grid
        self.one_SE_rule = one_SE_rule
        self.delta_theta_thresh = delta_theta_thresh
        self.theta_max_iter = theta_max_iter
        self.regressor_class = regressor_class
        self.theta_init = theta_init
        self.cv = cv
        self.warm_start_alpha = warm_start_alpha
        self.parallel = parallel
        if regressor_args is None:
            regressor_args = {}
        sub_params = remove_prefix_from_kwargs(regressor_args, "regressor")
        self.regressor_args = dict(sub_params)  # just for pylance  # just for pylance
        self.iterative_theta_regressor = None

    def update_regressor_args(self, theta):
        return {**self.regressor_args, "family": set_family_arg(theta)}

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ):

        # set initial theta

        regressor_args = self.update_regressor_args(self.theta_init)

        # fit CV with initial regressor
        logger.debug(f"Setting initial theta to {self.theta_init}")
        logger.debug(f"Launching CV")
        self.CV_regressor = CustomCVRegressor(
                                         param_grid=self.param_grid,
                                         one_SE_rule=self.one_SE_rule,
                                         regressor_class=self.regressor_class,
                                         cv = self.cv,
                                         warm_start_alpha=self.warm_start_alpha,
                                         parallel = self.parallel,
                                         **wrap_dict_with_prefix(self.regressor_args, "regressor"))

        self.CV_regressor.fit(X=X,
                         y = y,
                         sample_weight=sample_weight,
                         offset=offset)

        # set a iterative theta regressor with the best_regressor_params
        regressor_args = {**self.regressor_args, **self.CV_regressor.best_glm_params}
        logger.debug(f"Launching Iterative Theta on Best Params")

        if self.warm_start_alpha == True:
            # so that at each theta we do the full path
            alpha_path = regressor_args.pop("_alpha_path")
            regressor_class = WarmStartWrapper
            regressor_args_f = {
                "regressor_class": self.regressor_class,
                "alpha_param_key": self.CV_regressor.best_estimator.alpha_param_key,
                "start_params_key": self.CV_regressor.best_estimator.start_params_key,
                "alpha_path": alpha_path,
                **wrap_dict_with_prefix(regressor_args, "regressor"),
            }
            family_arg_name = "regressor__family"
        else:
            regressor_class = self.regressor_class
            regressor_args_f = regressor_args
            family_arg_name = "family"

        self.iterative_theta_regressor = IterativeThetaGLM(
            delta_theta_thresh=self.delta_theta_thresh,
            theta_max_iter=self.theta_max_iter,
            regressor_class=regressor_class,
            theta_init=self.theta_init,
            family_arg_name=family_arg_name,
            ** wrap_dict_with_prefix(regressor_args_f, "regressor"),
        )
        self.iterative_theta_regressor.fit(
            X=X, y=y, sample_weight=sample_weight, offset=offset
        )

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the regressor if not found
        on the wrapper itself.

        This is what makes predict / score / etc. transparently work.
        """
        # Avoid weirdness with special methods
        if name.startswith("__"):
            raise AttributeError(name)

        if self.iterative_theta_regressor is None:
            raise AttributeError(
                f"Attribute {name!r} not found on IterativeThetaAfterCVRegressor and "
                f"no inner iterative_theta_regressor is set yet (did you call .fit()?)."
            )

        return getattr(self.iterative_theta_regressor, name)
