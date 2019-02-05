#!/usr/bin/env python

import astrodata
import gemini_instruments
import numpy as np

from matplotlib import pyplot as plt
from copy import copy

from astropy.visualization import ImageNormalize, LinearStretch, \
    PercentileInterval
from astropy.wcs import WCS


filename = 'S20131121S0075_stack.fits'

palette = copy(plt.cm.viridis)
palette.set_bad('grey', 1.0)

ad = astrodata.open(filename)
hdr = ad[0].hdr
wcs = WCS(hdr)

masked_ad = np.ma.masked_where(ad[0].mask != 0, ad[0].data)

norm = ImageNormalize(masked_ad, stretch=LinearStretch(),
                      interval=PercentileInterval(95.))

fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
ax.imshow(masked_ad, origin='lower', vmin=norm.vmin, vmax=norm.vmax,
          cmap=palette)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
fig.savefig("{}.png".format(filename), dpi=192)
