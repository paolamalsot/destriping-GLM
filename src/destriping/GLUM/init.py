from types import SimpleNamespace
import pandas as pd
import numpy as np

def c_init_median_from_hw(df, h, w):
    # idea: initialize c (pd.Series with indices corresponding to cell_ids) to the median of corrected counts
    # here df["i"] or df["j"] values are converted to ints anyway
    assert h.index.dtype == int
    assert w.index.dtype == int
    corrected_counts = df["k"] / (
        h.get(df["i"].astype(int), 1.0).values * w.get(df["j"].astype(int), 1.0).values
    )
    c_init = corrected_counts.groupby(
        df["p"].astype(str)
    ).median()  # note that this will return an index of dtype object
    return c_init


def c_init_mean_from_hw(df, h, w):
    # idea: initialize c (pd.Series with indices corresponding to cell_ids) to the median of corrected counts
    # here df["i"] or df["j"] values are converted to ints anyway
    assert h.index.dtype == int
    assert w.index.dtype == int
    corrected_counts = df["k"] / (
        df["i"].map(h).fillna(1.0) * df["j"].map(w).fillna(1.0)
    )
    c_init = corrected_counts.groupby(
        df["p"].astype(str)
    ).mean()  # note that this will return an index of dtype object
    return c_init


def ratio_init_from_df(
    df: pd.DataFrame,
    epsilon: float = 0.0,
    quantile: float = 0.5,
    thresh_cell_counts: int = 10,
):
    """
    Compute row/col normalization factors (h, w) from a 'flat' obs-like DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'k', 'i', 'j', 'p'.
    epsilon : float
        Lower clip for factors.
    quantile : float
        Quantile to aggregate nuclear ratios per lane.
    thresh_cell_counts : int
        Cells with fewer than this many observations get NaN median and are ignored.

    Returns
    -------
    h : pd.Series
        Row ('i') factors indexed by unique i.
    w : pd.Series
        Column ('j') factors indexed by unique j.
    """
    # Work on a copy
    df_cp = df.copy()

    # Use a nucleus mask if provided; otherwise use all rows
    nucl_mask = ~pd.isna(df["p"])

    # Per-cell median and counts of n_counts
    cell_median = df_cp.groupby("p", observed=True)["k"].median()
    cell_counts = df_cp.groupby("p", observed=True)["k"].count()

    # Invalidate cells with too-few observations
    cell_median.loc[cell_counts < thresh_cell_counts] = np.nan

    # Broadcast per-row
    df_cp["cell_median"] = df_cp["p"].map(cell_median)
    df_cp["ratio"] = df_cp["k"] / df_cp["cell_median"]

    results = {}
    for factor_name, lane in [("h", "i"), ("w", "j")]:
        # Quantile over nuclear pixels by lane
        factor = (
            df_cp.loc[nucl_mask]
            .groupby(lane, observed=True)["ratio"]
            .quantile(q=quantile)
        )

        # Normalize to mean 1 over available lanes
        factor = factor / factor.mean()

        # Reindex to all lanes present in the data; fill missing with 1
        factor = factor.reindex(df_cp[lane].unique()).fillna(1.0)

        # Clip
        factor = factor.clip(lower=epsilon)

        results[factor_name] = factor

    return results["h"], results["w"]

def ones_init_from_df(df):
    i = df["i"].unique()
    di = len(i)
    j = df["j"].unique()
    dj = len(j)
    h = pd.Series(index = i, data = np.ones(di))
    w = pd.Series(index=j, data=np.ones(dj))
    return h, w


def quantiles_init_from_df(df, quantile, epsilon = 0):
    h = df.groupby("i", observed=True)["k"].quantile(quantile).clip(lower=epsilon)
    w = df.groupby("j", observed=True)["k"].quantile(quantile).clip(lower=epsilon)
    return h, w


def c_init_from_hw(df, h, w, c_mean = False):
    if c_mean:
        c = c_init_mean_from_hw(df, h, w)
    else:
        c = c_init_median_from_hw(df, h, w)
    return c

def get_ratio_init_sol(df, c_mean=False):
    h, w = ratio_init_from_df(df)
    c = c_init_from_hw(df, h, w, c_mean)
    return SimpleNamespace(h=h, w=w, c=c)

def get_ones_init_sol(df, c_mean =True):
    h, w = ones_init_from_df(df)
    c = c_init_from_hw(df, h, w, c_mean)
    return SimpleNamespace(h=h, w=w, c=c)

def get_quantiles_init_sol(df, quant, c_mean = True):
    h, w = quantiles_init_from_df(df, quant)
    c = c_init_from_hw(df, h, w, c_mean)
    return SimpleNamespace(h=h, w=w, c=c)
