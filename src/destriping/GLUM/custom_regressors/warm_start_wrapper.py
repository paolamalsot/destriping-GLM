import numpy as np
from copy import deepcopy
from src.destriping.GLUM.custom_regressors.helpers import extract_intercept_coef
from src.destriping.GLUM.custom_regressors.helpers import delegate_getattr
from src.destriping.GLUM.custom_regressors.helpers import (
    remove_prefix_from_kwargs,
    wrap_dict_with_prefix,
)
import logging
logger = logging.getLogger("WarmStartWrapper")
from src.destriping.GLUM.glum_nb_helpers import hash

class WarmStartWrapper:
    """
    Meta-estimator that runs a warm-started alpha path for a given (regressor_class, params) set.

    Parameters
    ----------

    regressor_class : object
    
    params : dict
        All hyperparameters to be passed at init to regressor_class
        (can contain the initial start_coef, and a dummy alpha)

    alpha_param_key : str
        For ex., 'alpha', 'regressor__alpha' in case of a nested regressor

    alpha_path : sequence of float
        Alpha values to traverse, in the exact order you want, typically
        from largest -> smallest.
    """

    def __init__(
        self,
        regressor_class,
        alpha_param_key: str,
        start_params_key: str,
        alpha_path,
        **params,
    ):
        self.regressor_class = regressor_class
        if params is None:
            params = {}
        sub_params = remove_prefix_from_kwargs(params, "regressor")
        self.params = dict(sub_params)  # just for pylance
        self.regressor = None
        self.alpha_param_key = alpha_param_key
        self.start_params_key = start_params_key
        self.alpha_path = list(alpha_path)

        if not self.alpha_path:
            raise ValueError("alpha_path must be a non-empty sequence.")

        # Will hold the final fitted estimator after .fit()
        self.estimator_ = None

        logger.debug(f"Initializing WarmStartWrapper for:\n class {self.regressor_class}\n {params=}\n {alpha_path=}")

    def _make_estimator_for_alpha(self, alpha, start_params):
        """
        Construct a regressor_class instance, set alpha and (optionally)
        start_params.
        """

        # this will use the start value in self.params if start_params is None
        params = {**self.params, self.alpha_param_key: alpha}    
        if start_params is not None:
            params[self.start_params_key] = start_params
            logger.debug(
                f"Initializing regressor of class {self.regressor_class} with start_param (hash = {hash(start_params)})"
            )

        est = self.regressor_class(**params)

        return est

    def fit(self, X, y, *, offset=None, sample_weight=None):
        """
        Fit along the alpha_path, re-instantiating the base estimator at
        each alpha and warm-starting via start_params.

        After this method, self.estimator_ is the estimator fitted
        at the last alpha in alpha_path.
        """
        warm_start = None
        est = None

        for alpha in self.alpha_path:
            logger.debug(f"Setting {alpha=}")
            est = self._make_estimator_for_alpha(alpha, warm_start)
            est.fit(X, y, sample_weight=sample_weight, offset=offset)

            warm_start = extract_intercept_coef(est)# Prepare warm start for next alpha
            logger.debug(f"Storing param (hash = {hash(warm_start)} for alpha {alpha})")

        self.estimator_ = est
        return self

    __getattr__ = delegate_getattr("estimator_", "regressor_class")
