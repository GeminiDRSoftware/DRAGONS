#!/usr/bin/env python
"""
Script created to generate a PNG file of the final stack produced during the
Flamingos-2 Tutorial
"""

import os
from copy import copy

import numpy as np
from astropy import visualization, wcs
from matplotlib import colors
from matplotlib import pyplot as plt

import astrodata


def main():

    filename = get_stack_filename()

    ad = astrodata.from_file(filename)

    fig = plt.figure(num=filename, figsize=(7, 4.5))
    fig.suptitle(os.path.basename(filename), y=0.97)

    axs = fig.subplots(1, len(ad), sharey=True)

    palette = copy(plt.cm.viridis)
    palette.set_bad("Gainsboro", 1.0)

    norm = visualization.ImageNormalize(
        np.dstack([ext.data for ext in ad]),
        stretch=visualization.LinearStretch(),
        interval=visualization.ZScaleInterval()
    )

    print(norm.vmin)
    print(norm.vmax)
    for i in range(len(ad)):

        axs[i].imshow(
            # np.ma.masked_where(ad[i].mask > 0, ad[i].data),
            ad[i].data,
            #norm=colors.Normalize(vmin=norm.vmin, vmax=norm.vmax),
            norm=colors.Normalize(vmin=750, vmax=900),
            origin="lower",
            cmap=palette,
        )

        axs[i].set_xlabel('d{:02d}'.format(i+1))
        axs[i].set_xticks([])

    axs[i].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 1], w_pad=0.05)

    fig.savefig(os.path.basename(filename.replace('.fits', '.png')))
    plt.show()


def get_stack_filename():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str,
                        help='Path to the stack fits file')

    args = parser.parse_args()

    return args.filename


if __name__ == main():
    main()
