import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors


letters_range = np.concatenate([np.arange(65, 91), np.arange(97, 123)])
letters = np.array([chr(i) for i in letters_range])  # 52 letters


# to map any N to letters range
def base_repr(x, base):
    # takes number and returns list
    d = x // base
    if d == 0:
        return [x]
    else:
        r = x % base
        return base_repr(d, base) + [r]


def int_to_word(x: np.ndarray):
    res = []
    for b in x.flatten():
        if b == -1:
            res.append(np.nan)
            continue
        else:
            list_str = [letters[y] for y in base_repr(b, 52)]
            res.append("".join(list_str))
    res = np.array(res, dtype=object).reshape(x.shape)
    res[x < 0] = np.nan
    return res


def int_to_str(x: np.ndarray):
    # converts int, with -1 meaning unassigned to str, with NaN meaning unassigned
    x = x.astype("str").astype(object)
    x[x == "-1"] = np.nan
    return x


def convert_str_to_rgb(x: np.ndarray, palette_="tab20", color_nan="black"):
    unique_values, int_val = np.unique(x.astype(str), return_inverse=True)
    len_unique_values = len(unique_values)
    palette = sns.color_palette(palette_, len_unique_values)
    colors = np.array(palette)[int_val]
    y = colors.reshape((*x.shape, colors.shape[-1]))
    # replace nan by black
    black_rgba = mcolors.to_rgb(color_nan)
    y[pd.isna(x)] = black_rgba

    return y, palette


def occupied_selector(series: pd.Series):
    return series.notna()
