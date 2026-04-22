from scanpy import read_10x_h5
import anndata
from h5py import File
import logging
import json
from matplotlib.image import imread
from pathlib import Path
import pandas as pd
import os
from src.spatialAdata.coords_orientation_convention import reorient_coords
from src.utilities.utilities import *
from src.spatialAdata.spatialAdata import spatialAdata
from src.utilities.df_unique_keys import img_df, coordinates_df


def load_visium_hd(path, source_image_path, count_file="filtered_feature_bc_matrix.h5"):
    """
    Creates a spatialAdata object from reading the folder and the source image path.
    Very similar to bin2cell's read_visium function

    path: path to directory for visium datafiles.
    source image path: path to the high-resolution tissue image
    count_file
        Which file in the passed directory to use as the count file. Typically would be one of:
        'filtered_feature_bc_matrix.h5' or 'raw_feature_bc_matrix.h5'
    """
    path = Path(path)
    adata = read_10x_h5(path / count_file, genome=None)  # counts matrix

    # getting the library_id
    with File(path / count_file, mode="r") as f:
        attrs = dict(f.attrs)

        library_id_list = attrs.pop("library_ids")
        library_id = str(library_id_list[0], "utf-8")

        logging.debug(f"{library_id_list=}")
        logging.debug(f"{library_id=}")

    tissue_positions_file = (
        path / "spatial/tissue_positions.csv"
        if (path / "spatial/tissue_positions.csv").exists()
        else (
            path / "spatial/tissue_positions.parquet"
            if (path / "spatial/tissue_positions.parquet").exists()
            else path / "spatial/tissue_positions_list.csv"
        )
    )

    files = dict(
        tissue_positions_file=tissue_positions_file,
        scalefactors_json_file=path / "spatial/scalefactors_json.json",
        hires_image=path / "spatial/tissue_hires_image.png",
        lowres_image=path / "spatial/tissue_lowres_image.png",
    )

    for f in files.values():
        if not f.exists():
            if any(x in str(f) for x in ["hires_image", "lowres_image"]):
                logging.warning(
                    f"You seem to be missing an image file.\n" f"Could not find '{f}'."
                )
            else:
                raise OSError(f"Could not find '{f}'")
    adata.uns["spatial"] = {library_id: {}}
    adata.uns["spatial"][library_id]["images"] = dict()
    for res in ["hires", "lowres"]:
        try:
            adata.uns["spatial"][library_id]["images"][res] = imread(
                str(files[f"{res}_image"])
            )
        except Exception:
            raise OSError(f"Could not find '{res}_image'")

    # read json scalefactors
    adata.uns["spatial"][library_id]["scalefactors"] = json.loads(
        files["scalefactors_json_file"].read_bytes()
    )

    adata.uns["spatial"][library_id]["metadata"] = {
        k: (str(attrs[k], "utf-8") if isinstance(attrs[k], bytes) else attrs[k])
        for k in ("chemistry_description", "software_version")
        if k in attrs
    }

    # read coordinates
    if files["tissue_positions_file"].name.endswith(".csv"):
        positions = pd.read_csv(
            files["tissue_positions_file"],
            header=0 if tissue_positions_file.name == "tissue_positions.csv" else None,
            index_col=0,
        )
    elif files["tissue_positions_file"].name.endswith(".parquet"):
        positions = pd.read_parquet(files["tissue_positions_file"])
        # need to set the barcode to be the index
        positions.set_index("barcode", inplace=True)

    positions.columns = [
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_col_in_fullres",
        "pxl_row_in_fullres",
    ]

    adata.obs = adata.obs.join(positions, how="left")

    # fill the obsm with our first coordinates. The first coordinate denotes the vertical axis, and the second the horizonatal axis.

    adata.obsm["coords__fullres"] = adata.obs[
        [
            "pxl_col_in_fullres",
            "pxl_row_in_fullres",
        ]  # changed this; apparently there was a mistake ?
    ].to_numpy()  # instead of "spatial" which does not mean anything

    coords_array = adata.obs[
        [
            "array_row",
            "array_col",
        ]
    ].to_numpy()

    # convert it to left_to_right
    # max_array_col = max(np.max(coords_array[:,1]), 3349)
    # the reason for the max is that in the step where we produce a gex image to input to stardist, the border around probably influences the stardist output...
    # coords_array[:,1] = max_array_col - coords_array[:,1]
    coords_array = reorient_coords(coords_array, adata.obsm["coords__fullres"])
    adata.obsm["coords__array"] = coords_array
    adata.obs[["array_row", "array_col"]] = adata.obsm["coords__array"]

    adata.obs.drop(
        columns=["pxl_row_in_fullres", "pxl_col_in_fullres"],
        inplace=True,
    )

    # put image path in uns
    if source_image_path is not None:
        # get an absolute path
        source_image_path = str(Path(source_image_path).resolve())
        adata.uns["spatial"][library_id]["metadata"]["source_image_path"] = str(
            source_image_path
        )

    fullres_fullres_coordinates = {
        "coordinate_id": "fullres",
        "img_key": "fullres",
        "scalefactor": 1.0,
    }

    fullres_hires_coordinates = {
        "coordinate_id": "fullres",  # .obsm["coords__fullres"]
        "img_key": "hires",
        # indeed full_res are coordinates wrt full resolution image !
        "scalefactor": adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_hires_scalef"
        ],
    }

    fullres_lowres_coordinates = {
        "coordinate_id": "fullres",  # .obsm["coords__fullres"]
        "img_key": "lowres",
        # indeed full_res are coordinates wrt full resolution image !
        "scalefactor": adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_lowres_scalef"
        ],
    }

    spots_coordinates = {"coordinate_id": "array", "img_key": None, "scalefactor": 1.0}

    coordinates_df_ = coordinates_df.from_records(
        [
            fullres_fullres_coordinates,
            fullres_hires_coordinates,
            fullres_lowres_coordinates,
            spots_coordinates,
        ]
    )

    fullres_img = {
        "img_key": "fullres",
        "in_memory": False,
        "path": source_image_path,
        "scalefactor": 1.0,
    }
    hires_img = {
        "img_key": "hires",
        "in_memory": True,
        "path": None,
        "scalefactor": adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_hires_scalef"
        ],
    }
    lowres_img = {
        "img_key": "lowres",
        "in_memory": True,
        "path": None,
        "scalefactor": adata.uns["spatial"][library_id]["scalefactors"][
            "tissue_lowres_scalef"
        ],
    }

    img_df_ = img_df.from_records([fullres_img, hires_img, lowres_img])

    # make unique variable names
    adata.var_names_make_unique()

    return spatialAdata(
        adata, library_id, coordinate_df=coordinates_df_, img_df=img_df_
    )


def load_spatialAdata(input_dir):
    # Load the .h5ad file into the adata attribute
    adata = anndata.read_h5ad(os.path.join(input_dir, "adata.h5ad"))

    # Load the coordinate_df CSV file into the coordinate_df attribute
    coordinates_df_ = coordinates_df.load(os.path.join(input_dir, "coordinate_df.csv"))
    img_df_ = img_df.load(os.path.join(input_dir, "img_df.csv"))

    # Load the library_id from the data_list.json file
    with open(os.path.join(input_dir, "spatial_adata_attributes.json"), "r") as file:
        attributes = json.load(file)

    return spatialAdata(
        adata, coordinate_df=coordinates_df_, img_df=img_df_, **attributes
    )
