from src.destriping.GLUM.custom_regressors.helpers import remove_prefix_from_kwargs
from src.destriping.GLUM.cv import glm_cv
from src.destriping.GLUM.custom_regressors.helpers import RegressorTypeLike
from glum._glm import ArrayLike
from typing import Optional


class CustomCVRegressor():

    def __init__(
        self,
        param_grid,
        one_SE_rule,
        regressor_class: RegressorTypeLike,
        cv,
        warm_start_alpha: bool,
        parallel: bool,
        **regressor_args,
    ):
        self.regressor_class = regressor_class
        self.regressor_args = regressor_args
        self.param_grid = param_grid
        self.one_SE_rule = one_SE_rule
        self.cv = cv
        self.warm_start_alpha = warm_start_alpha
        self.parallel = parallel
        if regressor_args is None:
            regressor_args = {}
        sub_params = remove_prefix_from_kwargs(regressor_args, "regressor")
        self.regressor_args = dict(sub_params)  # just for pylance
        self.best_estimator = None

    def fit(
        self,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        sample_weight: Optional[ArrayLike] = None,
        offset: Optional[ArrayLike] = None,
    ):

        self.best_estimator, self.best_glm_params, self.best_score, self.results = glm_cv(
                X,
                y,
                {**self.regressor_args, "cv": self.cv},
                self.param_grid,
                offset=offset,
                sample_weight=sample_weight,
                one_SE_rule=self.one_SE_rule,
                regressor_class=self.regressor_class,
                warm_start_alpha=self.warm_start_alpha,
                parallel = self.parallel
        )
    
        self.set_fitted_params(self.best_glm_params)

    def set_fitted_params(self, params_dict):
        # to ensure consistency with sklearn and glumCV, we set the fitted hp to name_
        for name, val in params_dict.items():
            self.__setattr__(name + "_", val)

    def __getattr__(self, name: str):
        """
        Delegate attribute access to the regressor if not found
        on the wrapper itself.

        This is what makes predict / score / etc. transparently work.
        """
        # Avoid weirdness with special methods
        if name.startswith("__"):
            raise AttributeError(name)

        if self.best_estimator is None:
            raise AttributeError(
                f"Attribute {name!r} not found on CustomCVRegressor and "
                f"no inner {self.regressor_class} is set yet (did you call .fit()?)."
            )

        return getattr(self.best_estimator, name)
