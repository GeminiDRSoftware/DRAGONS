#!/usr/bin/env python
"""
Script created to generate a PNG file of the final stack produced during the
Flamingos-2 Tutorial
"""

import astrodata
import gemini_instruments
import numpy as np
import os

from astropy import visualization, wcs
from copy import copy
from matplotlib import pyplot as plt


def main():

    filename = get_stack_filename()

    ad = astrodata.from_file(filename)

    data = ad[0].data
    mask = ad[0].mask
    header = ad[0].hdr

    masked_data = np.ma.masked_where(mask, data, copy=True)

    palette = copy(plt.cm.viridis)
    palette.set_bad('gray')

    norm_factor = visualization.ImageNormalize(
        masked_data,
        stretch=visualization.LinearStretch(),
        interval=visualization.ZScaleInterval(),
    )

    fig, ax = plt.subplots(subplot_kw={'projection': wcs.WCS(header)})

    ax.imshow(masked_data,
              cmap=palette,
              vmin=norm_factor.vmin,
              vmax=norm_factor.vmax)

    ax.set_title(os.path.basename(filename))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.savefig(filename.replace('.fits', '.png'))
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
