import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def custom_imshow(
    matrix,
    axis: matplotlib.axes.Axes | None = None,
    cmap="gray",
    na_color="red",
    colorbar_on=True,
    **kwargs
) -> matplotlib.image.AxesImage:
    if axis is None:
        fig, axis = plt.subplots()
    cmap = matplotlib.colormaps.get_cmap(cmap).copy()
    cmap.set_bad(color=na_color)
    pos = axis.imshow(matrix, cmap=cmap, **kwargs)
    fig = axis.figure
    if colorbar_on:
        fig.colorbar(pos, ax=axis)
    return pos
