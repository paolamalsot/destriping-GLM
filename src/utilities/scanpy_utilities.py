import numpy as np

na_color = (1.0, 0.0, 0.0, 0.0)


def get_palette_from_labels_he_colors(adata):
    # labels_he_colors is usually created during call to sc.plot.
    colors = adata.uns["labels_he_colors"]
    unique_labels = np.unique(adata.obs.labels_he.dropna().astype(int))
    my_list = list(zip(unique_labels, colors))
    palette = {str(key): val for key, val in my_list}
    return palette
