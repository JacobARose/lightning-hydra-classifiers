"""
lightning_hydra_classifiers/utils/plot_utils.py


Created on: Tuesday, July 27th, 2021
Author: Jacob A Rose


"""

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from PIL import Image

__all__ = ["colorbar", "display_images"]


def colorbar(mappable):

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


######################################################



def display_images(
                   images: List[Image.Image],
                   labels: Optional[List[str]]=None,
                   max_images: int=32,
                   columns: int=5,
                   width: int=20,
                   height: int=12,
                   label_wrap_length: int=50,
                   label_font_size: int="medium") -> None:

    if len(images) > max_images:
        print(f"Showing {max_images} images of {len(images)}:")
        images=images[0:max_images]

    rows = int(len(images)/columns)
#     height = max(height, rows * height)
    plt.subplots(rows, columns, figsize=(width, height), sharex=True, sharey=True)
    for i, image in enumerate(images):

        plt.subplot(rows + 1, columns, i + 1)
        plt.imshow(image)
        ax = plt.gca()
        for label in ax.xaxis.get_ticklabels():
            label.set_visible(False)
        for label in ax.yaxis.get_ticklabels():
            label.set_visible(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        title=None
        if isinstance(labels, list):
            title = labels[i]
        elif hasattr(image, 'filename'):
            title=image.filename
            
        if title:
#             if title.endswith("/"): title = title[:-1]
            title=Path(title).stem
            title=textwrap.wrap(title, label_wrap_length)
            title="\n".join(title)
            plt.title(title, fontsize=label_font_size); 

    plt.subplots_adjust(left=None, bottom=0.05, right=None, top=0.9, wspace=0.0, hspace=0.2*rows) #wspace=0.05, hspace=0.1)
