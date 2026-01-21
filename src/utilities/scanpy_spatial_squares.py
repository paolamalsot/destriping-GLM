import importlib
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# Import the module
module = importlib.import_module('scanpy.plotting._tools.scatterplots')

# Manually update globals with all attributes from the module, including private ones
globals().update({k: v for k, v in vars(module).items()})

def squares(
    x, y, *, s, ax, marker=None, c="b", vmin=None, vmax=None, scale_factor=1.0, **kwargs
):
    """
    Taken from here: https://gist.github.com/syrte/592a062c562cd2a98a83
    Make a scatter plot of circles.
    Similar to pl.scatter, but the size of circles are in data scale.
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    pl.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (https://opensource.org/license/bsd-3-clause/)
    """

    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.
    if scale_factor != 1.0:
        x = x * scale_factor
        y = y * scale_factor
    zipped = np.broadcast(x, y, s)
    patches = [Rectangle((x_, y_), 2*s_, 2*s_) for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        collection.set_array(np.ma.masked_invalid(c))
        collection.set_clim(vmin, vmax)
    else:
        collection.set_facecolor(c)

    ax.add_collection(collection)

    return collection

def embedding(
    adata: AnnData,
    basis: str,
    *,
    color: str | Sequence[str] | None = None,
    mask_obs: NDArray[np.bool_] | str | None = None,
    gene_symbols: str | None = None,
    use_raw: bool | None = None,
    sort_order: bool = True,
    edges: bool = False,
    edges_width: float = 0.1,
    edges_color: str | Sequence[float] | Sequence[str] = "grey",
    neighbors_key: str | None = None,
    arrows: bool = False,
    arrows_kwds: Mapping[str, Any] | None = None,
    groups: str | Sequence[str] | None = None,
    components: str | Sequence[str] | None = None,
    dimensions: tuple[int, int] | Sequence[tuple[int, int]] | None = None,
    layer: str | None = None,
    projection: Literal["2d", "3d"] = "2d",
    scale_factor: float | None = None,
    color_map: Colormap | str | None = None,
    cmap: Colormap | str | None = None,
    palette: str | Sequence[str] | Cycler | None = None,
    na_color: ColorLike = "lightgray",
    na_in_legend: bool = True,
    size: float | Sequence[float] | None = None,
    frameon: bool | None = None,
    legend_fontsize: int | float | _FontSize | None = None,
    legend_fontweight: int | _FontWeight = "bold",
    legend_loc: str = "right margin",
    legend_fontoutline: int | None = None,
    colorbar_loc: str | None = "right",
    vmax: VBound | Sequence[VBound] | None = None,
    vmin: VBound | Sequence[VBound] | None = None,
    vcenter: VBound | Sequence[VBound] | None = None,
    norm: Normalize | Sequence[Normalize] | None = None,
    add_outline: bool | None = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    ncols: int = 4,
    hspace: float = 0.25,
    wspace: float | None = None,
    title: str | Sequence[str] | None = None,
    show: bool | None = None,
    save: bool | str | None = None,
    ax: Axes | None = None,
    return_fig: bool | None = None,
    marker: str | Sequence[str] = ".",
    **kwargs,
) -> Figure | Axes | list[Axes] | None:
    """\
    Scatter plot for user specified embedding basis (e.g. umap, pca, etc)

    Parameters
    ----------
    basis
        Name of the `obsm` basis to use.
    {adata_color_etc}
    {edges_arrows}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.
    """
    #####################
    # Argument handling #
    #####################

    check_projection(projection)
    sanitize_anndata(adata)

    basis_values = _get_basis(adata, basis)
    dimensions = _components_to_dimensions(
        components, dimensions, projection=projection, total_dims=basis_values.shape[1]
    )
    args_3d = dict(projection="3d") if projection == "3d" else {}

    # Checking the mask format and if used together with groups
    if groups is not None and mask_obs is not None:
        raise ValueError("Groups and mask arguments are incompatible.")
    if mask_obs is not None:
        mask_obs = _check_mask(adata, mask_obs, "obs")

    # Figure out if we're using raw
    if use_raw is None:
        # check if adata.raw is set
        use_raw = layer is None and adata.raw is not None
    if use_raw and layer is not None:
        raise ValueError(
            "Cannot use both a layer and the raw representation. Was passed:"
            f"use_raw={use_raw}, layer={layer}."
        )
    if use_raw and adata.raw is None:
        raise ValueError(
            "`use_raw` is set to True but AnnData object does not have raw. "
            "Please check."
        )

    if isinstance(groups, str):
        groups = [groups]

    # Color map
    if color_map is not None:
        if cmap is not None:
            raise ValueError("Cannot specify both `color_map` and `cmap`.")
        else:
            cmap = color_map
    cmap = copy(colormaps.get_cmap(cmap))
    cmap.set_bad(na_color)
    # Prevents warnings during legend creation
    na_color = colors.to_hex(na_color, keep_alpha=True)

    if "edgecolor" not in kwargs:
        # by default turn off edge color. Otherwise, for
        # very small sizes the edge will not reduce its size
        # (https://github.com/scverse/scanpy/issues/293)
        kwargs["edgecolor"] = "none"

    # Vectorized arguments

    # turn color into a python list
    color = [color] if isinstance(color, str) or color is None else list(color)

    # turn marker into a python list
    marker = [marker] if isinstance(marker, str) else list(marker)

    if title is not None:
        # turn title into a python list if not None
        title = [title] if isinstance(title, str) else list(title)

    # turn vmax and vmin into a sequence
    if isinstance(vmax, str) or not isinstance(vmax, cabc.Sequence):
        vmax = [vmax]
    if isinstance(vmin, str) or not isinstance(vmin, cabc.Sequence):
        vmin = [vmin]
    if isinstance(vcenter, str) or not isinstance(vcenter, cabc.Sequence):
        vcenter = [vcenter]
    if isinstance(norm, Normalize) or not isinstance(norm, cabc.Sequence):
        norm = [norm]

    # Size
    if "s" in kwargs and size is None:
        size = kwargs.pop("s")
    if size is not None:
        # check if size is any type of sequence, and if so
        # set as ndarray
        if (
            size is not None
            and isinstance(size, (cabc.Sequence, pd.Series, np.ndarray))
            and len(size) == adata.shape[0]
        ):
            size = np.array(size, dtype=float)
    else:
        size = 120000 / adata.shape[0]

    ##########
    # Layout #
    ##########
    # Most of the code is for the case when multiple plots are required

    if wspace is None:
        #  try to set a wspace that is not too large or too small given the
        #  current figure size
        wspace = 0.75 / rcParams["figure.figsize"][0] + 0.02

    if components is not None:
        color, dimensions = list(zip(*product(color, dimensions)))

    color, dimensions, marker = _broadcast_args(color, dimensions, marker)

    # 'color' is a list of names that want to be plotted.
    # Eg. ['Gene1', 'louvain', 'Gene2'].
    # component_list is a list of components [[0,1], [1,2]]
    if (
        not isinstance(color, str)
        and isinstance(color, cabc.Sequence)
        and len(color) > 1
    ) or len(dimensions) > 1:
        if ax is not None:
            raise ValueError(
                "Cannot specify `ax` when plotting multiple panels "
                "(each for a given value of 'color')."
            )

        # each plot needs to be its own panel
        fig, grid = _panel_grid(hspace, wspace, ncols, len(color))
    else:
        grid = None
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, **args_3d)

    ############
    # Plotting #
    ############
    axs = []

    # use itertools.product to make a plot for each color and for each component
    # For example if color=[gene1, gene2] and components=['1,2, '2,3'].
    # The plots are: [
    #     color=gene1, components=[1,2], color=gene1, components=[2,3],
    #     color=gene2, components = [1, 2], color=gene2, components=[2,3],
    # ]
    for count, (value_to_plot, dims) in enumerate(zip(color, dimensions)):
        kwargs_scatter = kwargs.copy()  # is potentially mutated for each plot
        color_source_vector = _get_color_source_vector(
            adata,
            value_to_plot,
            layer=layer,
            mask_obs=mask_obs,
            use_raw=use_raw,
            gene_symbols=gene_symbols,
            groups=groups,
        )
        color_vector, color_type = _color_vector(
            adata,
            value_to_plot,
            values=color_source_vector,
            palette=palette,
            na_color=na_color,
        )

        # Order points
        order = slice(None)
        if sort_order and value_to_plot is not None and color_type == "cont":
            # Higher values plotted on top, null values on bottom
            order = np.argsort(-color_vector, kind="stable")[::-1]
        elif sort_order and color_type == "cat":
            # Null points go on bottom
            order = np.argsort(~pd.isnull(color_source_vector), kind="stable")
        # Set orders
        if isinstance(size, np.ndarray):
            size = np.array(size)[order]
        color_source_vector = color_source_vector[order]
        color_vector = color_vector[order]
        coords = basis_values[:, dims][order, :]

        # if plotting multiple panels, get the ax from the grid spec
        # else use the ax value (either user given or created previously)
        if grid:
            ax = plt.subplot(grid[count], **args_3d)
            axs.append(ax)
        if not (settings._frameon if frameon is None else frameon):
            ax.axis("off")
        if title is None:
            if value_to_plot is not None:
                ax.set_title(value_to_plot)
            else:
                ax.set_title("")
        else:
            try:
                ax.set_title(title[count])
            except IndexError:
                logg.warning(
                    "The title list is shorter than the number of panels. "
                    "Using 'color' value instead for some plots."
                )
                ax.set_title(value_to_plot)

        if color_type == "cont":
            vmin_float, vmax_float, vcenter_float, norm_obj = _get_vboundnorm(
                vmin, vmax, vcenter, norm=norm, index=count, colors=color_vector
            )
            kwargs_scatter["norm"] = check_colornorm(
                vmin_float,
                vmax_float,
                vcenter_float,
                norm_obj,
            )
            kwargs_scatter["cmap"] = cmap

        # make the scatter plot
        if projection == "3d":
            cax = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                c=color_vector,
                rasterized=settings._vector_friendly,
                marker=marker[count],
                **kwargs_scatter,
            )
        else:
            scatter = (
                partial(ax.scatter, s=size, plotnonfinite=True)
                if scale_factor is None
                else partial(
                    squares, s=size, ax=ax, scale_factor=scale_factor
                )  # size in circles is radius
            )

            if add_outline:
                # the default outline is a black edge followed by a
                # thin white edged added around connected clusters.
                # To add an outline
                # three overlapping scatter plots are drawn:
                # First black dots with slightly larger size,
                # then, white dots a bit smaller, but still larger
                # than the final dots. Then the final dots are drawn
                # with some transparency.

                bg_width, gap_width = outline_width
                point = np.sqrt(size)
                gap_size = (point + (point * gap_width) * 2) ** 2
                bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2
                # the default black and white colors can be changes using
                # the contour_config parameter
                bg_color, gap_color = outline_color

                # remove edge from kwargs if present
                # because edge needs to be set to None
                kwargs_scatter["edgecolor"] = "none"
                # For points, if user did not set alpha, set alpha to 0.7
                kwargs_scatter.setdefault("alpha", 0.7)

                # remove alpha and color mapping for outline
                kwargs_outline = {
                    k: v
                    for k, v in kwargs.items()
                    if k not in {"alpha", "cmap", "norm"}
                }

                for s, c in [(bg_size, bg_color), (gap_size, gap_color)]:
                    ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        s=s,
                        c=c,
                        rasterized=settings._vector_friendly,
                        marker=marker[count],
                        **kwargs_outline,
                    )

            cax = scatter(
                coords[:, 0],
                coords[:, 1],
                c=color_vector,
                rasterized=settings._vector_friendly,
                marker=marker[count],
                **kwargs_scatter,
            )

        # remove y and x ticks
        ax.set_yticks([])
        ax.set_xticks([])
        if projection == "3d":
            ax.set_zticks([])

        # set default axis_labels
        name = _basis2name(basis)
        axis_labels = [name + str(d + 1) for d in dims]

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if projection == "3d":
            # shift the label closer to the axis
            ax.set_zlabel(axis_labels[2], labelpad=-7)
        ax.autoscale_view()

        if edges:
            _utils.plot_edges(
                ax, adata, basis, edges_width, edges_color, neighbors_key=neighbors_key
            )
        if arrows:
            _utils.plot_arrows(ax, adata, basis, arrows_kwds)

        if value_to_plot is None:
            # if only dots were plotted without an associated value
            # there is not need to plot a legend or a colorbar
            continue

        if legend_fontoutline is not None:
            path_effect = [
                patheffects.withStroke(linewidth=legend_fontoutline, foreground="w")
            ]
        else:
            path_effect = None

        # Adding legends
        if color_type == "cat":
            _add_categorical_legend(
                ax,
                color_source_vector,
                palette=_get_palette(adata, value_to_plot),
                scatter_array=coords,
                legend_loc=legend_loc,
                legend_fontweight=legend_fontweight,
                legend_fontsize=legend_fontsize,
                legend_fontoutline=path_effect,
                na_color=na_color,
                na_in_legend=na_in_legend,
                multi_panel=bool(grid),
            )
        elif colorbar_loc is not None:
            plt.colorbar(
                cax, ax=ax, pad=0.01, fraction=0.08, aspect=30, location=colorbar_loc
            )

    if return_fig is True:
        return fig
    axs = axs if grid else ax
    _utils.savefig_or_show(basis, show=show, save=save)
    show = settings.autoshow if show is None else show
    if show:
        return None
    return axs

def spatial(
    adata: AnnData,
    *,
    basis: str = "spatial",
    img: np.ndarray | None = None,
    img_key: str | None | Empty = _empty,
    library_id: str | None | Empty = _empty,
    crop_coord: tuple[int, int, int, int] | None = None,
    alpha_img: float = 1.0,
    bw: bool | None = False,
    size: float = 1.0,
    scale_factor: float | None = None,
    spot_size: float | None = None,
    na_color: ColorLike | None = None,
    show: bool | None = None,
    return_fig: bool | None = None,
    save: bool | str | None = None,
    **kwargs,
) -> Figure | Axes | list[Axes] | None:
    """\
    Scatter plot in spatial coordinates.

    This function allows overlaying data on top of images.
    Use the parameter `img_key` to see the image in the background
    And the parameter `library_id` to select the image.
    By default, `'hires'` and `'lowres'` are attempted.

    Use `crop_coord`, `alpha_img`, and `bw` to control how it is displayed.
    Use `size` to scale the size of the Visium spots plotted on top.

    As this function is designed to for imaging data, there are two key assumptions
    about how coordinates are handled:

    1. The origin (e.g `(0, 0)`) is at the top left – as is common convention
    with image data.

    2. Coordinates are in the pixel space of the source image, so an equal
    aspect ratio is assumed.

    If your anndata object has a `"spatial"` entry in `.uns`, the `img_key`
    and `library_id` parameters to find values for `img`, `scale_factor`,
    and `spot_size` arguments. Alternatively, these values be passed directly.

    Parameters
    ----------
    {adata_color_etc}
    {scatter_spatial}
    {scatter_bulk}
    {show_save_ax}

    Returns
    -------
    If `show==False` a :class:`~matplotlib.axes.Axes` or a list of it.

    Examples
    --------
    This function behaves very similarly to other embedding plots like
    :func:`~scanpy.pl.umap`

    >>> import scanpy as sc
    >>> adata = sc.datasets.visium_sge("Targeted_Visium_Human_Glioblastoma_Pan_Cancer")
    >>> sc.pp.calculate_qc_metrics(adata, inplace=True)
    >>> sc.pl.spatial(adata, color="log1p_n_genes_by_counts")

    See Also
    --------
    :func:`scanpy.datasets.visium_sge`
        Example visium data.
    :doc:`/tutorials/spatial/basic-analysis`
        Tutorial on spatial analysis.
    """
    # get default image params if available
    library_id, spatial_data = _check_spatial_data(adata.uns, library_id)
    img, img_key = _check_img(spatial_data, img, img_key, bw=bw)
    spot_size = _check_spot_size(spatial_data, spot_size)
    scale_factor = _check_scale_factor(
        spatial_data, img_key=img_key, scale_factor=scale_factor
    )
    crop_coord = _check_crop_coord(crop_coord, scale_factor)
    na_color = _check_na_color(na_color, img=img)

    if bw:
        cmap_img = "gray"
    else:
        cmap_img = None
    circle_radius = size * scale_factor * spot_size * 0.5

    axs = embedding(
        adata,
        basis=basis,
        scale_factor=scale_factor,
        size=circle_radius,
        na_color=na_color,
        show=False,
        save=False,
        **kwargs,
    )
    if not isinstance(axs, list):
        axs = [axs]
    for ax in axs:
        cur_coords = np.concatenate([ax.get_xlim(), ax.get_ylim()])
        if img is not None:
            ax.imshow(img, cmap=cmap_img, alpha=alpha_img)
        else:
            ax.set_aspect("equal")
            #ax.invert_yaxis()
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(max(y_min, y_max), min(y_min, y_max))
        if crop_coord is not None:
            ax.set_xlim(crop_coord[0], crop_coord[1])
            ax.set_ylim(crop_coord[3], crop_coord[2])
        else:
            ax.set_xlim(cur_coords[0], cur_coords[1])
            ax.set_ylim(max(cur_coords[3], cur_coords[2]), min(cur_coords[3], cur_coords[2]))
    _utils.savefig_or_show("show", show=show, save=save)
    if return_fig:
        return axs[0].figure
    show = settings.autoshow if show is None else show
    if show:
        return None
    return axs