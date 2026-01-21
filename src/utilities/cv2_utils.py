import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

from PIL import Image

# setting needed so PIL can load the large TIFFs
Image.MAX_IMAGE_PIXELS = None

# setting needed so cv2 can load the large TIFFs
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2
from cv2 import imread, imwrite, cvtColor, COLOR_RGB2BGR


def save_tif_img_with_colorbar(img, save_path, log1p=False):
    if log1p:
        img = np.log1p(img)

    # make the values span from 0 to 255
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    img = (255 * (img - vmin) / (vmax - vmin)).astype(np.uint8)
    img = img.astype(np.uint8)

    # save or return image
    if save_path is not None:
        save_img(img, save_path, rgb=False)

        # saving the colorbar
        colorbar_path = save_path.replace(".tif", "_colorbar.tif")
        save_tif_colorbar(vmin, vmax, log1p, colorbar_path)


def save_img(img, img_path, rgb=True):
    if rgb:
        imwrite(img_path, cvtColor(img, COLOR_RGB2BGR))
    else:
        imwrite(img_path, img)


def load_img(img_path, rgb=True, dtype=np.uint8):
    # strongly inspired by bin2cell
    # TODO: verify: is there transposition done by cv2 (compared to imshow for example)
    img = imread(img_path)
    img = cvtColor(img, cv2.COLOR_BGR2RGB)
    if not (rgb):
        img = cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.astype(dtype, copy=False)


def save_tif_colorbar(vmin, vmax, log1p, colorbar_path):
    # vmin and vmax correspond to the min and max of the normalization white VS black...
    # log1p: whether the data was logged prior to normalization

    # Create and save the colorbar
    fig, ax = plt.subplots(figsize=(1, 5))  # Vertical colorbar
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap="gray"), cax=ax)

    tick_values = np.linspace(vmin, vmax, num=5)
    cb.set_ticks(tick_values)
    # Adjust ticks for log scale
    if log1p:
        ticklabels = [f"{np.expm1(t):.2f}" for t in tick_values]
    else:
        ticklabels = [f"{t:.2f}" for t in tick_values]

    cb.set_ticklabels(ticklabels)

    cb.ax.tick_params(labelsize=10)  # Optional: adjust tick font size
    # plt.axis('off')  # Remove extra axes
    plt.tight_layout()
    # Save the colorbar as a .tif file
    plt.savefig(
        colorbar_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True
    )
    plt.close(fig)
