from IPython.display import display, Markdown
import scipy
import numpy as np


def pretty(d, indent=0):
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print("\t" * (indent + 1) + f"array with shape: {value.shape}")
        elif isinstance(value, list):
            print("\t" * (indent + 1) + f"list with length: {len(value)}")
        elif isinstance(value, float):
            print("\t" * (indent + 1) + f"{value:.2f}")
        else:
            print("\t" * (indent + 1) + str(value))


def show_adata(adata):
    display(Markdown("## general"))
    print(adata)
    display(Markdown("## obs"))
    display(adata.obs.head(n=2))
    display(Markdown("## var"))
    display(adata.var.head(n=2))
    display(Markdown("## obsm"))
    display(adata.obsm)
    for key in adata.obsm.keys():
        print(f"{key}: array with shape{adata.obsm[key].shape}")
    display(Markdown("## uns"))
    pretty(adata.uns)


def img_2D_from_vals(array_coords, vals):
    # returns a img as a 2D numpy array
    max_0 = np.max(array_coords[:, 0])
    max_1 = np.max(array_coords[:, 1])
    min_0 = np.min(array_coords[:, 0])
    min_1 = np.min(array_coords[:, 1])
    dtype = vals.dtype if np.issubdtype(vals.dtype, np.floating) else object
    img = np.full((max_0 + 1 - min_0, max_1 + 1 - min_1), np.nan, dtype=dtype)
    img[array_coords[:, 0] - min_0, array_coords[:, 1] - min_1] = vals
    return img


def img_2D_to_vals(array_coords, img):
    min_0 = np.min(array_coords[:, 0])
    min_1 = np.min(array_coords[:, 1])
    pos_0_on_img = array_coords[:, 0] - min_0
    pos_1_on_img = array_coords[:, 1] - min_1
    return img[pos_0_on_img, pos_1_on_img]


def pull_values_adata(label, adata):
    # pull out the values for the image. start by checking .obs (makes a copy)
    if label in adata.obs.columns:
        vals = adata.obs[label].values

    elif label in adata.var_names:
        # if not in obs, it's presumably in the feature space
        vals = adata[:, label].X
        # may be sparse
        if scipy.sparse.issparse(vals):
            vals = vals.todense()
        # turn it to a flattened numpy array so it plays nice
        vals = np.asarray(vals).flatten()
    else:
        # failed to find
        raise ValueError('"' + label + '" not located in ``.obs`` or ``.var_names``')

    return vals.copy()


def get_item(adata, index):
    # returns a view of adata
    # Check if the index is a tuple (obj[index1, index2])
    if isinstance(index, tuple):
        if len(index) == 2:
            index1, index2 = index  # Unpack the tuple into two indices

            # Apply the two indices to the underlying object
            adata = adata[
                index1, index2
            ]  # Assuming your underlying object supports this operation

        else:
            raise IndexError("Too many indices. Expecting one or two indices.")

    # Handle the case of single indexing (obj[index])
    else:
        adata = adata[index]  # Single index

    return adata
