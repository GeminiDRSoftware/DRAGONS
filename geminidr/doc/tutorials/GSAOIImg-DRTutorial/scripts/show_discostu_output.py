#!/usr/bin/env python
"""
Reads a single-extensions file containing GSAOI data stacked and processed with DRAGONS
to be used in the GSAOIIMG-Tutorial.
"""

import astrodata
import gemini_instruments
import numpy as np

from astropy import visualization
from astropy.visualization import ImageNormalize, ZScaleInterval, LinearStretch
from astropy.wcs import WCS
from copy import copy
from matplotlib import pyplot as plt
from matplotlib import colors


def main():
    filename = get_filename()

    ad = astrodata.from_file(filename)
    print(ad.info())

    fig = plt.figure(num=filename, figsize=(8, 8))
    fig.suptitle('{}'.format(filename))

    palette = copy(plt.cm.viridis)
    palette.set_bad('w', 1.0)

    norm = ImageNormalize(np.dstack([ad[i].data for i in range(len(ad))]),
                          stretch=LinearStretch(),
                          interval=ZScaleInterval())

    wcs = WCS(ad[0].hdr)

    ax1 = fig.add_subplot(111, projection=wcs)
    ax1.imshow(np.ma.masked_where(ad[0].mask > 0, ad[0].data),
               norm=colors.Normalize(vmin=norm.vmin, vmax=norm.vmax),
               origin='lower', cmap=palette)

    ax1.set_xlabel('RA')
    ax1.set_ylabel('DEC')

    fig.tight_layout(rect=[0.15, 0.05, 1, 0.95])

    plt.savefig(filename.replace('.fits', '.png'))
    plt.show()


def get_filename():
    """
    Gets filename from command line
    """
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str,
                        help='Path to the fits file')

    args = parser.parse_args()

    return args.filename


if __name__ == main():
    main()
