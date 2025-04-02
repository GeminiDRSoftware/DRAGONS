#!/usr/bin/env python
"""
Displays a Flat Corrected and Sky Subtracted GSAOI data reduced with DRAGONS
to be used in the GSAOIIMG-Tutorial.
"""

import astrodata
import gemini_instruments
import numpy as np

from astropy import visualization
from copy import copy
from matplotlib import pyplot as plt
from matplotlib import colors


def main():

    # filename = 'S20170505S0102_flatCorrected.fits'
    filename = get_filename()

    ad = astrodata.from_file(filename)
    print(ad.info())

    fig = plt.figure(num=filename, figsize=(8, 8))
    fig.suptitle('{}'.format(filename))

    palette = copy(plt.cm.viridis)
    palette.set_bad('w', 1.0)

    norm = visualization.ImageNormalize(
        np.dstack([ad[i].data for i in range(4)]),
        stretch=visualization.LinearStretch(),
        interval=visualization.ZScaleInterval()
    )

    ax1 = fig.add_subplot(224)
    ax1.imshow(
        np.ma.masked_where(ad[0].mask > 0, ad[0].data),
        norm=colors.Normalize(vmin=norm.vmin, vmax=norm.vmax),
        origin='lower',
        cmap=palette
    )

    ax1.annotate('d1', (20, 20), color='white')
    ax1.set_xlabel('x [pixels]')
    ax1.set_ylabel('y [pixels]')

    ax2 = fig.add_subplot(223)
    ax2.imshow(
        np.ma.masked_where(ad[1].mask > 0, ad[1].data),
        norm=colors.Normalize(vmin=norm.vmin, vmax=norm.vmax),
        origin='lower',
        cmap=palette
    )

    ax2.annotate('d2', (20, 20), color='white')
    ax2.set_xlabel('x [pixels]')
    ax2.set_ylabel('y [pixels]')

    ax3 = fig.add_subplot(221)
    ax3.imshow(
        np.ma.masked_where(ad[2].mask > 0, ad[2].data),
        norm=colors.Normalize(vmin=norm.vmin, vmax=norm.vmax),
        origin='lower',
        cmap=palette
    )

    ax3.annotate('d3', (20, 20), color='white')
    ax3.set_xlabel('x [pixels]')
    ax3.set_ylabel('y [pixels]')

    ax4 = fig.add_subplot(222)
    ax4.imshow(
        np.ma.masked_where(ad[3].mask > 0, ad[3].data),
        norm=colors.Normalize(vmin=norm.vmin, vmax=norm.vmax),
        origin='lower',
        cmap=palette
    )

    ax4.annotate('d4', (20, 20), color='white')
    ax4.set_xlabel('x [pixels]')
    ax4.set_ylabel('y [pixels]')

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(filename.replace('.fits', '.png'))
    plt.show()


def get_filename():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str,
                        help='Path to the fits file')

    args = parser.parse_args()

    return args.filename


if __name__ == main():
    main()
