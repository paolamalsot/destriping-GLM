from src.destriping.GLUM.fit import categorizer_glum
import numpy as np


def P_hw_only_from_df(df, freeze_c=False, drop_first=True):
    _, _, categories_dict = categorizer_glum(df, freeze_c)
    offset_drop = 1 if drop_first else 0
    if freeze_c:
        P = np.concatenate(
            [
                np.ones(len(categories_dict["i"]) - offset_drop),
                np.ones(len(categories_dict["j"]) - offset_drop),
            ]
        )
    else:
        P = np.concatenate(
            [
                np.zeros(len(categories_dict["p"]) - offset_drop),
                np.ones(len(categories_dict["i"]) - offset_drop),
                np.ones(len(categories_dict["j"]) - offset_drop),
            ]
        )
    return P


def glum_default_alpha_range(n_alphas, alpha_max=10.0, min_alpha_ratio=1e-6):
    return np.geomspace(alpha_max, alpha_max * min_alpha_ratio, n_alphas)
