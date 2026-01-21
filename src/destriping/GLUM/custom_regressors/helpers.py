from typing import TypeAlias
from typing import Type, TypeAlias
from glum import GeneralizedLinearRegressor
from glum._glm import GeneralizedLinearRegressorBase
from typing import NamedTuple, Optional, Union, cast
from glum._glm import ArrayLike
import numpy as np

RegressorTypeLike: TypeAlias = (
    type["GeneralizedLinearRegressor"]
    | type["CustomCVRegressor"]
    | type["IterativeThetaGLM"]
)


def wrap_dict_with_prefix(d, prefix):
    """Return a new flat dict where each key is prefixed with `${prefix}__`."""
    return {f"{prefix}__{k}": v for k, v in d.items()}


def split_kwargs_by_prefix(kwargs, prefix):
    own = {}
    nested = {}
    full_prefix = prefix + "__"
    for k, v in kwargs.items():
        if k.startswith(full_prefix):
            nested[k[len(full_prefix) :]] = v
        else:
            own[k] = v
    return own, nested


def remove_prefix_from_kwargs(kwargs, prefix):
    own, nested = split_kwargs_by_prefix(kwargs, prefix)
    assert not (own), f"Python dict not empty: {own}"
    return nested


def extract_intercept_coef(glm):
    if getattr(glm, "fit_intercept", True):
        start_params_next = np.concatenate(
            ([glm.intercept_], np.asarray(glm.coef_).ravel())
        )
    else:
        start_params_next = np.asarray(glm.coef_).ravel()
    return start_params_next


def delegate_getattr(name_self_regressor, name_regressor_class):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)

        reg = getattr(self, name_self_regressor)
        if reg is None:
            cls = getattr(self, name_regressor_class)
            raise AttributeError(
                f"Attribute {name!r} not found on {type(self).__name__} "
                f"and no inner {cls} is set yet (did you call .fit()?)."
            )
        return getattr(reg, name)

    return __getattr__
