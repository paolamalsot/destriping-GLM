import pandas as pd
import os
from types import SimpleNamespace


def mu_from_sol(sol, df):
    return df["i"].map(sol.h) * df["j"].map(sol.w) * df["p"].map(sol.c)


# def get_mu_from_sol(sol, df):
#     return (
#         sol.h.get(df["i"].astype(int), 1.0).values
#         * sol.w.get(df["j"].astype(int), 1.0).values
#         * sol.c.get(df["p"]).values
#     )


def load_poisson_sol_hw_ns(poisson_sol_path):
    path = os.path.join(poisson_sol_path, "h.csv")
    h = pd.read_csv(path, index_col=0).iloc[:, 0]
    path = os.path.join(poisson_sol_path, "w.csv")
    w = pd.read_csv(path, index_col=0).iloc[:, 0]
    sol = SimpleNamespace(h=h, w=w)
    return sol
