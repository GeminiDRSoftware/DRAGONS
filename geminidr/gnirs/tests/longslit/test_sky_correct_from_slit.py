#!/usr/bin/env python3
"""
Tests for `skyCorrectFromSlit` for GNIRS longslit.
"""

from copy import deepcopy
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import astrodata
import geminidr
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

# -- Test parameters ----------------------------------------------------------
parameters = {'aperture_growth': 2,
              'lsigma': 3.0,
              'hsigma': 3.0,
              'niter': 3,
              'grow': 2}

# -- Test files ---------------------------------------------------------------
files = ['N20180114S0121_stack.fits', # LongRed
         'N20111231S0352_stack.fits', # ShortRed
         'N20170601S0291_stack.fits', # ShortBlue
         'N20220706S0306_stack.fits'] # LongBlue

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('function', ['spline3', 'chebyshev'])
@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('file', files)
def test_sky_correct_from_slit(file, order, function, change_working_dir,
                               path_to_inputs):

    ad = astrodata.open(os.path.join(path_to_inputs, file))

    p = GNIRSLongslit([deepcopy(ad)])
    ad_out = p.skyCorrectFromSlit(order=order, function=function, **parameters)[0]

    div = abs(ad[0].data / ad_out[0].data)

    # Check that ratio of the max values hasn't changed by more than 10%.
    assert (ad[0].data.max() / ad_out[0].data.max()) == pytest.approx(1, 1e-1)
    # Check that fewer than 10% of the pixels have changed by a factor of more
    # than 10.
    assert len(div[div>10]) / div.size * 100 < 10
    # Check that the average value of the background is within 0.01 sigma of 0.
    one_sigma = np.nanstd(ad_out[0].data)
    assert np.nanmedian(ad_out[0].data) < one_sigma * 0.01
    assert np.nanpercentile(ad_out[0].data, 50) < one_sigma * 0.01
