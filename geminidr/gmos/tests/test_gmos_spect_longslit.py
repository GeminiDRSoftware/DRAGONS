#!/usr/bin/env python
"""
Tests for GMOS Spect LS Extraction.

Notes
-----

    For extraction tests, your input wants to be a 2D image with an `APERTURE`
    table attached. You'll see what happens if you take a spectrophotometric
    standard and run it through the standard reduction recipe, but the
    `APERTURE` table has one row per aperture with the following columns:

    - number : sequential list of aperture number

    - ndim, degree, domain_start, domain_end, c0, [c1, c2, c3...] : standard
    Chebyshev1D definition of the aperture centre (in pixels) as a function of
    pixel in the dispersion direction

    - aper_lower : location of bottom of aperture relative to centre (always
    negative)

    - aper_upper : location of top of aperture relative to centre (always
    positive)

    The ndim column will always be 1 since it's always 1D Chebyshev, but the
    `model_to_dict()` and `dict_to_model()` functions that convert the Model
    instance to a dict create/require this.
"""

import os
import pytest

import astrofaker
import astrodata
import geminidr
import numpy as np

from astropy import table
from astropy.io import fits
from astrodata import testing
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from geminidr.gmos import primitives_gmos_spect
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

try:
    import astrofaker

    HAS_ASTROFAKER = True
except ImportError:
    HAS_ASTROFAKER = False

# ToDo @bquint: These files are not used for now but I am keeping them for future regression tests
test_cases = [

    # GMOS-N B600 at 0.600 um ---
    ('GMOS/GN-2018A-Q-302-56', [
        'N20180304S0121.fits',  # Standard
        'N20180304S0122.fits',  # Standard
        'N20180304S0123.fits',  # Standard
        'N20180304S0124.fits',  # Standard
        'N20180304S0125.fits',  # Standard
        'N20180304S0126.fits',  # Standard
        'N20180304S0204.fits',  # Bias
        'N20180304S0205.fits',  # Bias
        'N20180304S0206.fits',  # Bias
        'N20180304S0207.fits',  # Bias
        'N20180304S0208.fits',  # Bias
        'N20180304S0122.fits',  # Flat
        'N20180304S0123.fits',  # Flat
        'N20180304S0126.fits',  # Flat
        'N20180302S0397.fits',  # Arc
    ]),

]


@pytest.mark.skipif("not HAS_ASTROFAKER")
def test_find_apertures():

    data = np.zeros((100, 200))
    data[50] = 10.

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = data

    ad = astrofaker.create('GMOS-S')
    ad.add_extension(hdu, pixel_scale=1.0)

    _p = primitives_gmos_spect.GMOSSpect([ad])
    _p.findSourceApertures()


@pytest.mark.skipif("not HAS_ASTROFAKER")
def test_extract_1d_spectra():

    data = np.zeros((100, 200))
    data[50] = 10.

    aperture = table.Table(
        [[1],  # Number
         [1],  # ndim
         [0],  # degree
         [0],  # domain_start
         [data.shape[1] - 1],  # domain_end
         [50],  # c0
         [-3],  # aper_lower
         [3],  # aper_upper
         ],
        names=[
            'number',
            'ndim',
            'degree',
            'domain_start',
            'domain_end',
            'c0',
            'aper_lower',
            'aper_upper'],
    )

    hdu = fits.ImageHDU()
    hdu.header['CCDSUM'] = "1 1"
    hdu.data = data

    ad = astrofaker.create('GMOS-S')
    ad.add_extension(hdu, pixel_scale=1.0)
    ad[0].APERTURE = aperture

    _p = primitives_gmos_spect.GMOSSpect([])
    ade = _p.extract1DSpectra(ad)[0]

    np.testing.assert_equal(ade[0].shape[0], data.shape[1])
    np.testing.assert_equal(ade[0].data, data[50])




if __name__ == '__main__':
    pytest.main()
