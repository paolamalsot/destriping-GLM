from copy import deepcopy
from src.destriping.GLUM.fit import fit_GLM_glum
from src.destriping.GLUM.init import get_ratio_init_sol, get_ones_init_sol, get_quantiles_init_sol
from src.destriping.GLUM.sdata_to_df import format_obs_to_df
from types import SimpleNamespace
from src.destriping.GLUM.cv_splits import default_cv_splits, cv_splits
from src.destriping.GLUM.penalties import P_hw_only_from_df
from src.destriping.GLUM.penalties import glum_default_alpha_range
from glum._glm import PoissonDistribution, NegativeBinomialDistribution
from src.destriping.GLUM.custom_regressors.iterative_theta_regressor import (
    fit_theta_iter_status_keys,
)
from src.destriping.utils.make_sol import make_sol
from typing import Callable

init_methods = ["median_ratio", "quantiles", "ones", None]
cv_split_method = ["default", "spatial"]

def initialize_penalty(P, df, freeze_c):
    if P == "hw_only":
        P = P_hw_only_from_df(df, freeze_c=freeze_c)
    return P


def build_poisson_sol_from_stripings(sol, spatialdata, cell_id_label):
    """
    Wrap (h,w,c) into a PoissonSol aligned to spatialdata nucleus indices.
    """
    sol_new = deepcopy(sol)
    sol_new.c.index = sol_new.c.index.to_series().apply(lambda x: x.replace("id_", "",1))
    sol_new_c = (
        spatialdata.obs.loc[spatialdata.nucl_indices(cell_id_label), cell_id_label]
        .map(sol_new.c)
        .astype(float)
    )
    # sol_new_c = sol_new.c.reindex(
    #    spatialdata.obs.loc[spatialdata.nucl_indices(cell_id_label), cell_id_label]
    # )
    sol_new.c = sol_new_c
    nucl_idx = spatialdata.nucl_indices(cell_id_label)

    # Build PoissonSol
    poisson_sol = make_sol(
        sol.h.index,
        sol.h.values,
        sol.w.index,
        sol.w.values,
        spatialdata.obs.loc[nucl_idx, "array_row"].values,
        spatialdata.obs.loc[nucl_idx, "array_col"].values,
        nucl_idx,
        sol_new.c.loc[nucl_idx].values,
    )
    return poisson_sol

class GlumWrapper():

    def __init__(
        self,
        data,
        id_label,
        init_method: str | None,
        init_method_args = None,
        family = "nbinom",
        family_params = None,
        n_alphas = None,
        alphas = None,
        P2 = "hw_only",
        P1 = "hw_only",
        cv_split_method="default",
        freeze_c=False,
        regressorCV=False,
        sklearnCV=False,
        sklearnCV_one_SE_rule=True,
        warm_start_alpha=False,
        fit_theta_iter=False,
        fit_theta_max_iter=3,
        fit_theta_iter_loc="in",
        timeout=None,
        **kwargs,
    ):  
        # timeout for compatibility; doesn't do anything
        self.family = family
        if family_params is None:
            family_params = {}
        self.family_params = family_params
        if init_method_args is None:
            init_method_args = {}
        self.init_method_args = init_method_args
        self.n_alphas = n_alphas
        self.alphas = alphas
        self.P2 = P2
        self.P1 = P1
        self.data = data
        self.cv_split_method = cv_split_method
        self.id_label = id_label
        self.init_method = init_method
        self.freeze_c = freeze_c
        self.regressorCV = regressorCV
        self.sklearnCV = sklearnCV
        self.sklearnCV_one_SE_rule = sklearnCV_one_SE_rule
        self.warm_start_alpha = warm_start_alpha
        self.fit_theta_iter = fit_theta_iter
        self.fit_theta_max_iter = fit_theta_max_iter
        self.fit_theta_iter_loc = fit_theta_iter_loc
        self.timeout = timeout
        self.kwargs = kwargs
        self.glm_ = None

    def df_from_sdata(self):
        self.data.add_array_coords_to_obs()
        obs = self.data.obs
        obs = obs.query(
            f"~{self.id_label}.isna()"
            ).copy()  # this will NOT erase the cells "NA" "NaN" ?
        obs[self.id_label] = obs[self.id_label].astype(str)
        obs[self.id_label] = "id_" + obs[self.id_label]
        self.df = format_obs_to_df(obs, self.id_label)

    def initialize_P2(self):
        return initialize_penalty(self.P2, self.df, self.freeze_c)

    def initialize_P1(self):
        return initialize_penalty(self.P1, self.df, self.freeze_c)

    def initialize_sol(self):
        args = {"c_mean": True, **self.init_method_args}
        match self.init_method:
            case "median_ratio":
                init_sol = get_ratio_init_sol(self.df, **args)
            case "ones":
                init_sol = get_ones_init_sol(self.df, **args)
            case "quantiles":
                init_sol = get_quantiles_init_sol(self.df, **args)
            case None:
                init_sol = SimpleNamespace(h = None, w = None, c = None)
            case _:
                raise ValueError("Wrong init method")

        self.h_start = init_sol.h
        self.w_start = init_sol.w
        self.c_start = init_sol.c

    def make_splits(self):
        match self.cv_split_method:
            case "default":
                cv = default_cv_splits(self.df)
            case "spatial":
                cv = cv_splits(self.df)
            case _:
                raise ValueError("Wrong cv split method")
        return cv

    def init_family(self):
        if self.family == "poisson":
            family_cls = PoissonDistribution
        elif self.family == "nbinom":
            family_cls = NegativeBinomialDistribution
        else:
            raise ValueError("Wrong family name")
        return family_cls(**self.family_params)

    def initialize_alphas(self):
        if self.alphas is None:
            alphas = glum_default_alpha_range(n_alphas=self.n_alphas)
        else:
            alphas = self.alphas
        return alphas

    def fit(self):

        self.df_from_sdata()
        self.initialize_sol()

        if self.regressorCV or self.sklearnCV:
            cv = self.make_splits()
            P2 = self.initialize_P2()
            P1 = self.initialize_P1()
            alphas = self.initialize_alphas()
            supp_kwargs = {"cv": cv,
                           "P2": P2,
                           "P1": P1,
                           "alphas": alphas}
        else:
            supp_kwargs = {}

        family = self.init_family()

        h_hat, w_hat, c_hat, self.glm_ = fit_GLM_glum(
            df = self.df,
            h_start = self.h_start,
            w_start = self.w_start,
            c_start = self.c_start,
            freeze_c = self.freeze_c,
            regressorCV = self.regressorCV,
            sklearnCV = self.sklearnCV,
            sklearnCV_one_SE_rule = self.sklearnCV_one_SE_rule,
            warm_start_alpha = self.warm_start_alpha,
            fit_theta_iter = self.fit_theta_iter,
            fit_theta_max_iter = self.fit_theta_max_iter,
            fit_theta_iter_loc = self.fit_theta_iter_loc,
            family = family,
            **self.kwargs,
            **supp_kwargs
        )
        self.sol_ = SimpleNamespace(h=h_hat, w=w_hat, c=c_hat)

    def get_sol(self):
        poisson_sol = build_poisson_sol_from_stripings(self.sol_, self.data, self.id_label)
        if self.family == "nbinom":
            dist = "nbinom"
            dist_params = {"r": self.glm_.family.theta}
        elif self.family == "poisson":
            dist = "poisson"
            dist_params = {}
        else:
            raise ValueError("Unknown family")
        return poisson_sol, dist, dist_params

    @property
    def status_dict(self):
        dict_ = {}
        if self.fit_theta_max_iter:
            for key in fit_theta_iter_status_keys:
                dict_[key] = getattr(self.glm_, key)
        dict_["converged"] = (self.glm_.n_iter_ < self.glm_.max_iter)
        return dict_

    @property
    def alpha(self):
        return self.glm_.alpha

    @property
    def theta(self):
        return self.glm_.theta
