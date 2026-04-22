import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _validate_labels(df: pd.DataFrame, cell_id_label: str):
    labels = df[cell_id_label]
    if not isinstance(labels.dtype, pd.CategoricalDtype):
        raise TypeError(
            f"Column '{cell_id_label}' must be category dtype, got {labels.dtype}. "
            "Cast it with df[cell_id_label].astype('category') before calling this function."
        )
    return labels


def _extract_unique_adjacent_pairs(
    df: pd.DataFrame,
    cell_id_label: str,
) -> tuple[np.ndarray, pd.Index, pd.Series]:
    """Return unique original cell-cell adjacency pairs as integer codes.

    Uses 8-connected bin adjacency:
      - horizontal
      - vertical
      - both diagonals
    """
    labels = _validate_labels(df, cell_id_label)

    codes = labels.cat.codes
    categories = labels.cat.categories

    df_work = df[["array_row", "array_col"]].assign(_code=codes.to_numpy())
    cols_needed = ["array_row", "array_col", "_code"]

    pair_frames = []

    # Offsets chosen canonically so each neighbor relation is visited once
    neighbor_offsets = [
        (0, 1),  # right
        (1, 0),  # down
        (1, 1),  # down-right
        (1, -1),  # down-left
    ]

    for d_row, d_col in neighbor_offsets:
        df_shifted = df_work.assign(
            array_row=df_work["array_row"] + d_row,
            array_col=df_work["array_col"] + d_col,
        )
        pairs = (
            df_work[cols_needed]
            .merge(
                df_shifted[cols_needed].rename(columns={"_code": "_code_r"}),
                on=["array_row", "array_col"],
            )
            .rename(columns={"_code": "_code_l"})
        )
        pair_frames.append(pairs)

    all_bin_pairs = pd.concat(pair_frames, ignore_index=True)

    mask = (
        (all_bin_pairs["_code_l"] >= 0)
        & (all_bin_pairs["_code_r"] >= 0)
        & (all_bin_pairs["_code_l"] != all_bin_pairs["_code_r"])
    )

    bin_pairs = all_bin_pairs.loc[mask, ["_code_l", "_code_r"]].to_numpy(dtype=np.int32)

    if len(bin_pairs) == 0:
        unique_pairs = np.empty((0, 2), dtype=np.int32)
    else:
        bin_pairs = np.sort(bin_pairs, axis=1)
        unique_pairs = np.unique(bin_pairs, axis=0)

    return unique_pairs, categories, codes


def _labels_from_fused_pairs(
    fused_pairs: np.ndarray,
    n_cells: int,
    categories: pd.Index,
    codes: pd.Series,
    out_index: pd.Index,
) -> pd.Series:
    """Build output categorical labels from selected fused cell-cell pairs."""
    if len(fused_pairs) > 0:
        r = np.concatenate([fused_pairs[:, 0], fused_pairs[:, 1]])
        c = np.concatenate([fused_pairs[:, 1], fused_pairs[:, 0]])
        adj = coo_matrix(
            (np.ones(len(r), dtype=np.int8), (r, c)),
            shape=(n_cells, n_cells),
        )
        _, comp_labels = connected_components(adj, directed=False)
    else:
        comp_labels = np.arange(n_cells, dtype=np.int32)

    # Representative = smallest code in each component
    order = np.argsort(comp_labels, kind="stable")
    sorted_comp = comp_labels[order]
    first_idx = np.r_[0, np.flatnonzero(np.diff(sorted_comp)) + 1]
    reps = order[
        first_idx
    ]  # smallest original code in each component because order sorted by code

    comp_to_repr = np.empty(comp_labels.max() + 1, dtype=np.int32)
    comp_to_repr[sorted_comp[first_idx]] = reps

    repr_code = comp_to_repr[comp_labels]

    code_to_repr_id = pd.Series(categories[repr_code], index=pd.RangeIndex(n_cells))
    new_labels = codes.map(code_to_repr_id)
    new_labels.index = out_index
    return new_labels.astype(pd.CategoricalDtype(categories=categories))


def fuse_neighbouring_cells(
    df: pd.DataFrame,
    cell_id_label: str,
    p: float,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Standalone fusion at probability p on the original adjacency graph."""
    if not (0 <= p <= 1):
        raise ValueError(f"p must be in [0, 1], got {p}")

    if rng is None:
        rng = np.random.default_rng()

    unique_pairs, categories, codes = _extract_unique_adjacent_pairs(df, cell_id_label)
    n_cells = len(categories)

    keep = rng.random(len(unique_pairs)) < p
    fused_pairs = unique_pairs[keep]

    return _labels_from_fused_pairs(
        fused_pairs=fused_pairs,
        n_cells=n_cells,
        categories=categories,
        codes=codes,
        out_index=df.index,
    )


def fuse_neighbouring_cells_nested(
    df: pd.DataFrame,
    cell_id_label: str,
    ps: list[float],
    rng: np.random.Generator | None = None,
) -> dict[float, pd.Series]:
    """
    Monotone nested fusion across p values using one shared random draw per
    original adjacent cell pair.

    For each original unique neighboring pair e, sample u_e ~ Uniform(0,1) once.
    At threshold p, fuse e iff u_e < p.

    This ensures:
      - results are nested in p
      - each fixed p matches the standalone Bernoulli(p) edge model marginally
    """
    if rng is None:
        rng = np.random.default_rng()

    ps_sorted = sorted(ps)
    if any((p < 0 or p > 1) for p in ps_sorted):
        raise ValueError("All p values must be in [0, 1]")

    unique_pairs, categories, codes = _extract_unique_adjacent_pairs(df, cell_id_label)
    n_cells = len(categories)

    edge_u = rng.random(len(unique_pairs))

    result: dict[float, pd.Series] = {}
    for p in ps_sorted:
        fused_pairs = unique_pairs[edge_u < p]
        result[p] = _labels_from_fused_pairs(
            fused_pairs=fused_pairs,
            n_cells=n_cells,
            categories=categories,
            codes=codes,
            out_index=df.index,
        )

    return result
