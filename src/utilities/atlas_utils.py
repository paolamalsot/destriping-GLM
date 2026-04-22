import json
import re

import numpy as np
import pandas as pd


def load_deepslice_anchoring(predictions_json_path):
    """Load DeepSlice anchoring vectors and image dimensions from predictions.json."""
    with open(predictions_json_path) as f:
        data = json.load(f)

    slice_data = data["slices"][0]
    anchoring = slice_data["anchoring"]

    O = np.array(anchoring[0:3])
    U = np.array(anchoring[3:6])
    V = np.array(anchoring[6:9])
    width = slice_data["width"]
    height = slice_data["height"]

    return O, U, V, width, height


def pixel_to_atlas_coords(pixel_coords, O, U, V, img_width, img_height):
    """Convert pixel coordinates to atlas voxel coordinates.

    Parameters
    ----------
    pixel_coords : np.ndarray, shape (N, 2)
        Pixel coordinates in (row, col) order — i.e. (y, x).
    O, U, V : np.ndarray, shape (3,)
        DeepSlice anchoring vectors (origin, horizontal, vertical).
    img_width, img_height : int
        Image dimensions in pixels.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Atlas voxel coordinates.
    """
    nx = pixel_coords[:, 1] / img_width   # col 1 = columns (x) → normalize by width
    ny = pixel_coords[:, 0] / img_height  # col 0 = rows (y) → normalize by height
    atlas_coords = O[None, :] + nx[:, None] * U[None, :] + ny[:, None] * V[None, :]
    return atlas_coords


def extract_cortical_layer(acronym, is_isocortex=False):
    """Extract cortical layer from an Allen atlas region acronym.

    E.g. 'SSp-bfd2/3' → 'L2/3', 'MOp6a' → 'L6a'. Returns None for non-cortical regions.

    Parameters
    ----------
    acronym : str
        Allen atlas region acronym.
    is_isocortex : bool
        Whether the region is a descendant of Isocortex (ID 315). Only extracts
        layers for isocortex regions, avoiding false positives like CA1 → L1.
    """
    if not is_isocortex:
        return None
    m = re.search(r"(6[ab]|2/3|[1-5])$", acronym)
    return f"L{m.group()}" if m else None


def annotate_coords_with_atlas(
    pixel_coords, predictions_json_path, atlas_name="allen_mouse_25um"
):
    """Map pixel coordinates to brain region annotations via DeepSlice anchoring.

    Parameters
    ----------
    pixel_coords : np.ndarray, shape (N, 2)
        Pixel coordinates in (row, col) order — i.e. (y, x).
    predictions_json_path : str or Path
        Path to DeepSlice predictions.json.
    atlas_name : str
        BrainGlobe atlas name.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: atlas_region_id, atlas_region_acronym, atlas_region_name,
        cortical_layer (L1/L2/3/L4/L5/L6a/L6b or None).
    """
    from brainglobe_atlasapi import BrainGlobeAtlas

    O, U, V, width, height = load_deepslice_anchoring(predictions_json_path)
    atlas = BrainGlobeAtlas(atlas_name)
    annotation = atlas.annotation

    atlas_coords = pixel_to_atlas_coords(pixel_coords, O, U, V, width, height)
    voxel_indices = np.round(atlas_coords).astype(int)

    # Clip to atlas bounds
    for dim in range(3):
        voxel_indices[:, dim] = np.clip(
            voxel_indices[:, dim], 0, annotation.shape[dim] - 1
        )

    region_ids = annotation[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ]

    # Build lookup from structures
    ISOCORTEX_ID = 315
    acronyms = []
    names = []
    is_isocortex_flags = []
    for rid in region_ids:
        if rid == 0:
            acronyms.append("outside")
            names.append("Outside brain")
            is_isocortex_flags.append(False)
        else:
            structure = atlas.structures[rid]
            acronyms.append(structure["acronym"])
            names.append(structure["name"])
            is_isocortex_flags.append(ISOCORTEX_ID in structure["structure_id_path"])

    layers = [
        extract_cortical_layer(acr, is_iso)
        for acr, is_iso in zip(acronyms, is_isocortex_flags)
    ]

    return pd.DataFrame(
        {
            "atlas_region_id": region_ids,
            "atlas_region_acronym": acronyms,
            "atlas_region_name": names,
            "cortical_layer": layers,
        }
    )
