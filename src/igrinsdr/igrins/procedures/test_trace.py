import numpy as np
from trace_flat import identify_horizontal_line
from trace_flat import trace_centroids_chevyshev

import astropy.io.fits as pyfits
import scipy.ndimage as ni


if __name__ == '__main__':
    hdu = pyfits.open("SDCH_20220301_0011_lampstack.fits")
    tbl = trace_flat_edges(hdu[1].data)

