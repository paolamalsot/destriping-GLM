import numpy as np
import pandas as pd


def subsample_labels_nested(
    labels: pd.Series,
    percentages: list[float],
    seed: int,
) -> dict[float, pd.Series]:
    """Subsample cell labels in a nested fashion.

    At each percentage level the retained set is a strict subset of the set
    retained at the previous (higher) level, guaranteeing nesting.

    Parameters
    ----------
    labels:
        Per-bin cell IDs (``sdata.obs[cell_id_label]``).  NaN = background.
    percentages:
        Fractions of nuclei to keep, e.g. ``[0.95, 0.90, 0.80, 0.50]``.
        Need not be sorted; the function sorts them descending internally.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict mapping each percentage to a subsampled ``pd.Series`` with the same
    index as *labels*.  Cells not in the kept set are set to NaN.
    """
    rng = np.random.default_rng(seed)
    percentages_desc = sorted(percentages, reverse=True)

    all_cell_ids = np.array(labels.dropna().unique())
    n_total = len(all_cell_ids)

    current_pool = all_cell_ids.copy()
    result = {}

    for pct in percentages_desc:
        n_keep = int(round(pct * n_total))
        if n_keep > len(current_pool):
            n_keep = len(current_pool)
        kept = rng.choice(current_pool, size=n_keep, replace=False)
        kept_set = set(kept)
        subsampled = labels.where(labels.isin(kept_set), other=np.nan)
        result[pct] = subsampled
        current_pool = kept

    return result
