# experimenting by doing cv_splits by region of the slide -> 4 fold cross-validation
import numpy as np
from sklearn.model_selection._split import check_cv

def group_indices(df):
    """
    Splits the DataFrame into four groups based on the medians of columns 'i' and 'j'.
    Returns a dictionary of index arrays for each group.
    """
    median_i = df["i"].median()
    median_j = df["j"].median()

    split_1 = df.query("(i <= @median_i) and (j <= @median_j)").index
    split_2 = df.query("(i <= @median_i) and (j > @median_j)").index
    split_3 = df.query("(i > @median_i) and (j <= @median_j)").index
    split_4 = df.query("(i > @median_i) and (j > @median_j)").index
    return [df.index.get_indexer(s) for s in [split_1, split_2, split_3, split_4]]


def cv_splits(df):
    all_pos = np.arange(len(df))

    folds = group_indices(df)
    cv_splits = [
        (np.setdiff1d(all_pos, test_idx, assume_unique=False), test_idx)
        for test_idx in folds
        if len(test_idx) > 0
    ]
    return cv_splits

def default_cv_splits(df):
    y = np.arange(len(df))
    X = np.arange(len(df))
    # default used by glum, see line 550 of _glm_cv
    return list(check_cv().split(X, y))
