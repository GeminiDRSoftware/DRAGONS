#!/usr/bin/env python
"""
Script created to generate a PNG file of the final stack produced during the
Flamingos-2 Tutorial
"""

import os
from copy import copy

import numpy as np

from astropy import visualization, wcs
from matplotlib import pyplot as plt

import astrodata


def main():
    args = _parse_args()
    filename = args.filename

    ad = astrodata.from_file(filename)

    data = ad[0].data
    mask = ad[0].mask
    header = ad[0].hdr

    if args.mask:
        masked_data = np.ma.masked_where(mask, data, copy=True)
    else:
        masked_data = data

    palette = copy(plt.cm.viridis)
    palette.set_bad('Gainsboro')

    norm_factor = visualization.ImageNormalize(
        masked_data,
        stretch=visualization.LinearStretch(),
        interval=visualization.ZScaleInterval(),
    )

    fig = plt.figure(num=filename)
    ax = fig.subplots(subplot_kw={"projection": wcs.WCS(header)})

    print(norm_factor.vmin)
    print(norm_factor.vmax)
    ax.imshow(masked_data,
              cmap=palette,
              #vmin=norm_factor.vmin,
              #vmax=norm_factor.vmax,
              vmin=750., vmax=900.,
              origin='lower')

    ax.set_title(os.path.basename(filename))

    ax.coords[0].set_axislabel('Right Ascension')
    ax.coords[0].set_ticklabel(fontsize='small')

    ax.coords[1].set_axislabel('Declination')
    ax.coords[1].set_ticklabel(rotation='vertical', fontsize='small')

    fig.tight_layout(rect=[0.05, 0, 1, 1])
    fig.savefig(os.path.basename(filename.replace('.fits', '.png')))
    plt.show()


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str,
                        help='Path to the stack fits file')
    parser.add_argument('--mask', action='store_true',
                        help='Apply masked color')

    args = parser.parse_args()

    return args


if __name__ == main():
    main()
