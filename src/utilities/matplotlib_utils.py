from __future__ import annotations
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil
import uuid
import os
import numpy as np
from PIL import Image


def auto_figsize(nrows=1, ncols=1, base_width=4, base_height=3):
    return (base_width * ncols, base_height * nrows)


def subplots_autosize(nrows, ncols=1, base_width=4, base_height=3):
    # Example: 2 rows × 3 columns
    return plt.subplots(
        nrows, ncols, figsize=auto_figsize(nrows, ncols, base_width, base_height)
    )


def pad_axes_in_points(ax, pad_left=0, pad_right=0, pad_bottom=0, pad_top=0):
    """
    Add padding around axes in *points* (1/72 inch) on each side.
    Works with different x/y scales (log, asinh, etc.).
    """

    fig = ax.figure
    fig = ax.figure
    fig.canvas.draw()  # make sure transData is correct
    dpi = fig.dpi

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Convert each pad from points → pixels
    L = pad_left * dpi / 72.0
    R = pad_right * dpi / 72.0
    B = pad_bottom * dpi / 72.0
    T = pad_top * dpi / 72.0

    x_disp_max, y_disp_max = ax.transData.transform((xmax, ymax))
    x_disp_min, y_disp_min = ax.transData.transform((xmin, ymin))

    x_disp_max_new = x_disp_max + R
    y_disp_max_new = y_disp_max + T
    x_max_new, y_max_new = ax.transData.inverted().transform(
        (x_disp_max_new, y_disp_max_new)
    )
    x_disp_min_new = x_disp_min - L
    y_disp_min_new = y_disp_min - B
    x_min_new, y_min_new = ax.transData.inverted().transform(
        (x_disp_min_new, y_disp_min_new)
    )
    ax.set_xlim(x_min_new, x_max_new)
    ax.set_ylim(y_min_new, y_max_new)


import matplotlib.pyplot as plt


def break_axes(ax, xlim=None, ylim=None, *, gap=0.03, d=0.015, linewidth=1.0):
    """
    Replace `ax` with axes that simulate a broken x-axis, broken y-axis, or both.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to replace.
    xlim : None | (xmin, xmax) | ((xmin1, xmax1), (xmin2, xmax2))
        None -> keep existing xlim.
    ylim : None | (ymin, ymax) | ((ymin1, ymax1), (ymin2, ymax2))
        None -> keep existing ylim.
    gap : float
        Fraction of the original axes width/height used as whitespace gap.
    d : float
        Size of diagonal break marks in axes coordinates.
    linewidth : float
        Line width for break marks.

    Returns
    -------
    If no breaks: returns a single Axes.
    If broken y only: returns (ax_top, ax_bot)
    If broken x only: returns (ax_left, ax_right)
    If broken x and y: returns ((ax_tl, ax_tr), (ax_bl, ax_br))
    """

    def _is_pair(v):
        return isinstance(v, (tuple, list)) and len(v) == 2

    def _is_broken(v):
        # True if v looks like ((a,b),(c,d))
        return _is_pair(v) and _is_pair(v[0]) and _is_pair(v[1])

    def _span(r):
        a, b = r
        s = abs(b - a)
        return s if s != 0 else 1.0

    def _diag_y(ax_here, where):
        # where: "bottom" (on top axes) or "top" (on bottom axes)
        kw = dict(
            transform=ax_here.transAxes, color="k", clip_on=False, linewidth=linewidth
        )
        if where == "bottom":
            ax_here.plot((-d, +d), (-d, +d), **kw)
            ax_here.plot((1 - d, 1 + d), (-d, +d), **kw)
        elif where == "top":
            ax_here.plot((-d, +d), (1 - d, 1 + d), **kw)
            ax_here.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        else:
            raise ValueError("where must be 'bottom' or 'top'.")

    def _diag_x(ax_here, where):
        # where: "right" (on left axes) or "left" (on right axes)
        kw = dict(
            transform=ax_here.transAxes, color="k", clip_on=False, linewidth=linewidth
        )
        if where == "right":
            ax_here.plot((1 - d, 1 + d), (-d, +d), **kw)
            ax_here.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        elif where == "left":
            ax_here.plot((-d, +d), (-d, +d), **kw)
            ax_here.plot((-d, +d), (1 - d, 1 + d), **kw)
        else:
            raise ValueError("where must be 'right' or 'left'.")

    # Defaults: keep current limits if None
    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    break_x = _is_broken(xlim)
    break_y = _is_broken(ylim)

    fig = ax.figure
    pos = ax.get_position()

    # If neither axis is broken: just apply limits and return original ax
    if not break_x and not break_y:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return ax

    # Normalize ranges
    if break_x:
        (x_left, x_right) = xlim
    else:
        x_left = x_right = xlim  # same range everywhere

    if break_y:
        (y_bot, y_top) = ylim
    else:
        y_bot = y_top = ylim  # same range everywhere

    # Compute layout: 1 or 2 columns, 1 or 2 rows
    ncols = 2 if break_x else 1
    nrows = 2 if break_y else 1

    gap_w = pos.width * float(gap) if ncols == 2 else 0.0
    gap_h = pos.height * float(gap) if nrows == 2 else 0.0

    usable_w = pos.width - gap_w
    usable_h = pos.height - gap_h
    if usable_w <= 0 or usable_h <= 0:
        raise ValueError("gap is too large for the axes size.")

    # Width ratios based on x spans (only if broken x)
    if ncols == 2:
        w_left = usable_w * (_span(x_left) / (_span(x_left) + _span(x_right)))
        w_right = usable_w - w_left
    else:
        w_left = usable_w

    # Height ratios based on y spans (only if broken y)
    if nrows == 2:
        h_bot = usable_h * (_span(y_bot) / (_span(y_bot) + _span(y_top)))
        h_top = usable_h - h_bot
    else:
        h_bot = usable_h

    # Remove original
    ax.remove()

    axes = {}

    # Create axes
    if nrows == 1 and ncols == 2:
        # broken x only
        ax_l = fig.add_axes([pos.x0, pos.y0, w_left, usable_h])
        ax_r = fig.add_axes(
            [pos.x0 + w_left + gap_w, pos.y0, w_right, usable_h], sharey=ax_l
        )

        ax_l.set_xlim(*x_left)
        ax_r.set_xlim(*x_right)
        ax_l.set_ylim(*y_bot)
        ax_r.set_ylim(*y_bot)

        # spines + labels
        ax_l.spines["right"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.tick_params(labelleft=False)

        # break marks
        _diag_x(ax_l, "right")
        _diag_x(ax_r, "left")

        return ax_l, ax_r

    if nrows == 2 and ncols == 1:
        # broken y only
        ax_b = fig.add_axes([pos.x0, pos.y0, usable_w, h_bot])
        ax_t = fig.add_axes(
            [pos.x0, pos.y0 + h_bot + gap_h, usable_w, h_top], sharex=ax_b
        )

        ax_b.set_xlim(*x_left)
        ax_t.set_xlim(*x_left)
        ax_b.set_ylim(*y_bot)
        ax_t.set_ylim(*y_top)

        ax_t.spines["bottom"].set_visible(False)
        ax_b.spines["top"].set_visible(False)
        ax_t.tick_params(labelbottom=False)

        _diag_y(ax_t, "bottom")
        _diag_y(ax_b, "top")

        return ax_t, ax_b

    # broken x and y => 2x2
    # Bottom-left
    ax_bl = fig.add_axes([pos.x0, pos.y0, w_left, h_bot])
    # Bottom-right
    ax_br = fig.add_axes(
        [pos.x0 + w_left + gap_w, pos.y0, w_right, h_bot], sharey=ax_bl
    )
    # Top-left
    ax_tl = fig.add_axes([pos.x0, pos.y0 + h_bot + gap_h, w_left, h_top], sharex=ax_bl)
    # Top-right
    ax_tr = fig.add_axes(
        [pos.x0 + w_left + gap_w, pos.y0 + h_bot + gap_h, w_right, h_top],
        sharex=ax_br,
        sharey=ax_tl,
    )

    # Limits
    ax_bl.set_xlim(*x_left)
    ax_br.set_xlim(*x_right)
    ax_tl.set_xlim(*x_left)
    ax_tr.set_xlim(*x_right)

    ax_bl.set_ylim(*y_bot)
    ax_br.set_ylim(*y_bot)
    ax_tl.set_ylim(*y_top)
    ax_tr.set_ylim(*y_top)

    # Hide interior spines
    # y-break (between top and bottom)
    for a in (ax_tl, ax_tr):
        a.spines["bottom"].set_visible(False)
        a.tick_params(labelbottom=False)
    for a in (ax_bl, ax_br):
        a.spines["top"].set_visible(False)

    # x-break (between left and right)
    for a in (ax_tl, ax_bl):
        a.spines["right"].set_visible(False)
    for a in (ax_tr, ax_br):
        a.spines["left"].set_visible(False)
        a.tick_params(labelleft=False)

    # Break marks for y (top boundary of bottom axes, bottom boundary of top axes)
    _diag_y(ax_tl, "bottom")
    _diag_y(ax_tr, "bottom")
    _diag_y(ax_bl, "top")
    _diag_y(ax_br, "top")

    # Break marks for x (right boundary of left axes, left boundary of right axes)
    _diag_x(ax_tl, "right")
    _diag_x(ax_bl, "right")
    _diag_x(ax_tr, "left")
    _diag_x(ax_br, "left")

    return (ax_tl, ax_tr), (ax_bl, ax_br)


def save_plot_interior_tif(fig, ax, save_path, dpi=300):
    fig.set_dpi(dpi)

    fig.canvas.draw()

    # Full canvas as RGBA array
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)

    # Axes bbox in pixel coords
    bbox = ax.get_window_extent()
    x0, y0, x1, y1 = map(int, [bbox.x0, bbox.y0, bbox.x1, bbox.y1])

    # Note: buffer origin is top-left; Matplotlib display origin is bottom-left
    cropped = buf[h - y1 : h - y0, x0:x1, :]

    Image.fromarray(cropped).save(save_path, compression="tiff_lzw")


def save_pdf_temp_intermediate(
    temp_dir: str | os.PathLike | None, output_path: str | os.PathLike, **kwargs
) -> Path:
    """
    Save the *current* Matplotlib figure as a PDF by writing locally first,
    then copying/replacing into the final destination (e.g., iCloud folder).

    Parameters
    ----------
    temp_dir:
        Directory used for the intermediate local write. If None, uses OS temp dir.
        Will be created if it doesn't exist.
    output_path:
        Final PDF path (can be on iCloud). Parent directories will be created.

    Returns
    -------
    Path to the final saved PDF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pick / create temp dir
    if temp_dir is None:
        base_tmp = Path(tempfile.gettempdir())
    else:
        base_tmp = Path(temp_dir)
        base_tmp.mkdir(parents=True, exist_ok=True)

    # Unique intermediate file names to avoid collisions
    token = uuid.uuid4().hex
    local_pdf = base_tmp / f".matplotlib_save_{token}.pdf"
    tmp_dst = output_path.with_suffix(output_path.suffix + f".tmp_{token}")

    # 1) Save locally (fast, reliable)
    plt.savefig(local_pdf, **kwargs)

    # 2) Copy to destination, then atomic replace
    shutil.copy2(local_pdf, tmp_dst)
    tmp_dst.replace(output_path)

    # 3) Cleanup local intermediate
    try:
        local_pdf.unlink(missing_ok=True)
    except TypeError:  # py<3.8 compatibility, just in case
        if local_pdf.exists():
            local_pdf.unlink()

    return output_path
