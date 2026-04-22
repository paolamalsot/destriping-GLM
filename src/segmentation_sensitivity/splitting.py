import numpy as np
import pandas as pd


def _split_one_cell(grp: pd.DataFrame) -> pd.Series:
    """Return a boolean Series (True = child B) for one cell's group.

    Cuts along the midpoint of the longest bounding-box axis.  Bins strictly
    above the midpoint become child B; bins at or below stay in child A.
    Falls back to an index-based half-split for degenerate cells.
    """
    rows = grp["array_row"].to_numpy()
    cols = grp["array_col"].to_numpy()

    dx = int(cols.max()) - int(cols.min())
    dy = int(rows.max()) - int(rows.min())

    if dx >= dy:
        mid = (int(cols.min()) + int(cols.max())) / 2
        mask = cols > mid
    else:
        mid = (int(rows.min()) + int(rows.max())) / 2
        mask = rows > mid

    if mask.all() or not mask.any():
        mask = np.zeros(len(rows), dtype=bool)
        mask[len(rows) // 2:] = True

    return pd.Series(mask, index=grp.index)


def split_cells(
    df: pd.DataFrame,
    cell_id_label: str,
    p: float,
    min_size: int = 2,
    max_cells: int | None = None,
    n_jobs: int = 1,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Split cells stochastically to simulate over-segmentation errors.

    Each cell with >= min_size bins is split independently with probability p,
    using an axis-midpoint cut along the longest bounding-box axis.
    New child-B IDs are guaranteed not to collide with existing IDs.

    Parameters
    ----------
    df : DataFrame with columns array_row (int), array_col (int), and
         cell_id_label (category dtype; NA = background).
    cell_id_label : name of the cell ID column.
    p : probability of splitting each eligible cell.
    min_size : minimum bin count for a cell to be eligible.  Must be >= 2.
    max_cells : if set, cap the number of cells actually split (subsample from
        the eligible set using rng if more than max_cells are selected by p).
    n_jobs : ignored (kept for API compatibility).
    rng : NumPy random generator.  If None, a fresh default_rng() is used.

    Returns
    -------
    pd.Series with the same index as df and category dtype.
    New child-B categories are appended to the existing categories.

    Raises
    ------
    TypeError  : if df[cell_id_label] is not category dtype.
    ValueError : if min_size < 2.
    """
    labels = df[cell_id_label]
    if not pd.api.types.is_categorical_dtype(labels):
        raise TypeError(
            f"Column '{cell_id_label}' must be category dtype, got {labels.dtype}. "
            "Cast it with df[cell_id_label].astype('category') before calling this function."
        )
    if min_size < 2:
        raise ValueError(f"min_size must be >= 2, got {min_size}.")

    if rng is None:
        rng = np.random.default_rng()

    codes_np = labels.cat.codes.to_numpy()
    categories = labels.cat.categories
    n_cells = len(categories)

    cell_sizes = np.bincount(codes_np[codes_np >= 0], minlength=n_cells)

    eligible = np.where((cell_sizes >= min_size) & (rng.random(n_cells) < p))[0]
    if max_cells is not None and len(eligible) > max_cells:
        eligible = rng.choice(eligible, size=max_cells, replace=False)
    split_codes = eligible

    if len(split_codes) == 0:
        return labels.copy()

    # Collision-free IDs for child B: parent-derived so the same cell always gets
    # the same child ID regardless of which p value first selects it.
    if pd.api.types.is_integer_dtype(categories):
        base = int(categories.max()) + 1
        new_ids = [int(categories[code]) + base for code in split_codes]
    else:
        max_len = max(len(str(c)) for c in categories)
        new_ids = [
            f"{str(categories[code])}{'_' * (max_len - len(str(categories[code])) + 1)}C"
            for code in split_codes
        ]
    code_to_new_id = dict(zip(split_codes.tolist(), new_ids))

    # Sub-DataFrame for selected bins; group by integer code Series (not a column)
    sub = df.loc[np.isin(codes_np, split_codes), ["array_row", "array_col"]]
    sub_codes = pd.Series(codes_np, index=df.index).loc[sub.index]

    child_b_mask = sub.groupby(sub_codes, sort=False, group_keys=False).apply(
        _split_one_cell
    )

    # Vectorised assignment of new IDs
    child_b_idx = child_b_mask.index[child_b_mask.to_numpy()]
    out_ids = labels.astype(object).copy()
    out_ids.loc[child_b_idx] = sub_codes.loc[child_b_idx].map(code_to_new_id).to_numpy()

    all_categories = list(categories) + new_ids
    return out_ids.astype(pd.CategoricalDtype(categories=all_categories))


def split_cells_nested(
    df: pd.DataFrame,
    cell_id_label: str,
    ps: list[float],
    min_size: int = 2,
    max_cells: int | None = None,
    n_jobs: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[float, pd.Series]:
    """Split cells with nested selection sets across a list of p values.

    A single per-cell threshold is drawn once from U[0, 1].  A cell is selected
    for splitting at probability p if its threshold < p, so the selected sets are
    automatically nested: selected(p1) ⊆ selected(p2) whenever p1 < p2.  Each
    output is computed from the *original* labels (not chained), and a cell always
    receives the same axis-midpoint cut regardless of which p first selects it.

    The groupby split is computed once for all cells eligible at max(ps) and
    reused for all smaller p values.

    Parameters
    ----------
    df : DataFrame with columns ``array_row``, ``array_col``, and
         ``cell_id_label`` (category dtype; NA = background).
    cell_id_label : Name of the column containing cell IDs.
    ps : List of split probabilities.
    min_size : Minimum bin count for a cell to be eligible.  Must be >= 2.
    max_cells : If set, cap the number of cells actually split per p value.
    n_jobs : ignored (kept for API compatibility).
    rng : NumPy random generator.  If None, a fresh default_rng() is used.

    Returns
    -------
    dict mapping each p → pd.Series of split labels (same index as ``df``).
    """
    labels = df[cell_id_label]
    if not pd.api.types.is_categorical_dtype(labels):
        raise TypeError(
            f"Column '{cell_id_label}' must be category dtype, got {labels.dtype}. "
            "Cast it with df[cell_id_label].astype('category') before calling this function."
        )
    if min_size < 2:
        raise ValueError(f"min_size must be >= 2, got {min_size}.")

    if rng is None:
        rng = np.random.default_rng()

    codes_np = labels.cat.codes.to_numpy()
    categories = labels.cat.categories
    n_cells = len(categories)

    cell_sizes = np.bincount(codes_np[codes_np >= 0], minlength=n_cells)
    eligible_mask = cell_sizes >= min_size

    # Draw one threshold per cell upfront
    thresholds = rng.random(n_cells)

    # Run groupby.apply once for all cells eligible at max(ps)
    max_p = max(ps)
    all_selected = np.where(eligible_mask & (thresholds < max_p))[0]

    if len(all_selected) == 0:
        return {p: labels.copy() for p in ps}

    # Pre-generate new_ids for all potentially-split cells.
    # IDs are parent-derived so the same cell always gets the same child ID,
    # and all_new_ids is shared across every p value so the CategoricalDtype
    # category ordering is identical — ensuring consistent plot colours.
    if pd.api.types.is_integer_dtype(categories):
        base = int(categories.max()) + 1
        all_new_ids = [int(categories[code]) + base for code in all_selected]
    else:
        max_len = max(len(str(c)) for c in categories)
        all_new_ids = [
            f"{str(categories[code])}{'_' * (max_len - len(str(categories[code])) + 1)}C"
            for code in all_selected
        ]
    code_to_new_id_all = dict(zip(all_selected.tolist(), all_new_ids))
    all_categories_full = list(categories) + all_new_ids

    # Single groupby.apply over all potentially-split bins
    sub_all = df.loc[np.isin(codes_np, all_selected), ["array_row", "array_col"]]
    sub_codes_all = pd.Series(codes_np, index=df.index).loc[sub_all.index]

    child_b_mask_all = sub_all.groupby(
        sub_codes_all, sort=False, group_keys=False
    ).apply(_split_one_cell)

    # Pre-extract child-B index and their codes (reused across all p values)
    child_b_idx_all = child_b_mask_all.index[child_b_mask_all.to_numpy()]
    child_b_codes_all = sub_codes_all.loc[child_b_idx_all]

    result: dict[float, pd.Series] = {}

    for p in sorted(ps):
        selected_p = np.where(eligible_mask & (thresholds < p))[0]

        if max_cells is not None and len(selected_p) > max_cells:
            selected_p = rng.choice(selected_p, size=max_cells, replace=False)

        if len(selected_p) == 0:
            result[p] = labels.copy()
            continue

        code_to_new_id_p = {c: code_to_new_id_all[c] for c in selected_p.tolist()}

        # Filter pre-computed child-B to this p's selected cells
        in_p = np.isin(child_b_codes_all.to_numpy(), selected_p)
        child_b_idx_p = child_b_idx_all[in_p]

        out_ids = labels.astype(object).copy()
        out_ids.loc[child_b_idx_p] = (
            child_b_codes_all.iloc[in_p].map(code_to_new_id_p).to_numpy()
        )

        # Use the full category list (all potential new IDs from max_p) so that
        # every p value shares the same CategoricalDtype — consistent plot colours.
        result[p] = out_ids.astype(pd.CategoricalDtype(categories=all_categories_full))

    return result
