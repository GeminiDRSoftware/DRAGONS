# pytest suite
"""
Unit tests for :any:`geminidr.ghost.primitives_ghost`.

This is a suite of tests to be run with pytest.
"""
import pytest
from copy import deepcopy

import numpy as np

import astrodata, gemini_instruments
from geminidr.ghost.primitives_ghost import GHOST

from astropy.io import fits


BINNING_OPTIONS = [
    (1, 1,),
    (2, 1,),
    (2, 2,),
    (4, 1,),
    (4, 2,),
    (8, 1,),
]


@pytest.mark.ghostunit
@pytest.mark.parametrize("binning", BINNING_OPTIONS)
def test_rebin_ghost_ad(binning):
    """
    Checks to make:

    - Re-binned data arrays are the correct shape;
    - Correct keyword headers have been updated;

    """
    # Create a test data frame to operate on
    phu = fits.PrimaryHDU()
    phu.header['OBSERVAT'] = "GEMINI-SOUTH"
    hdu = fits.ImageHDU(data=np.zeros((1024, 1024,)), name='SCI')
    hdu.header['CCDSUM'] = '1 1'
    hdu.header['DATASEC'] = '[1:1024,1:1024]'
    hdu.header['TRIMSEC'] = '[1:1024,1:1024]'
    hdu.header['AMPSIZE'] = '[1:1024,1:1024]'
    ad = astrodata.create(phu, [hdu, ])

    # Rebin the data a bunch of different ways, run tests on each
    # re-binning

    # Re-bin the data
    ad_new = deepcopy(ad)
    p = GHOST(adinputs=[ad_new])
    ad_new = p._rebin_ghost_ad(ad_new, binning[0], binning[1])
    assert np.all([ad_new[0].data.shape[i] ==
                   (ad[0].data.shape[i] / binning[abs(i - 1)])
                   for i in [0, 1]]), 'Rebinned data has incorrect ' \
                                      'shape (expected {}, ' \
                                      'have {})'.format(
        tuple((ad[0].data.shape[i] / binning[abs(i - 1)])
              for i in [0, 1]), ad_new[0].data.shape,
    )
    assert ad_new[0].hdr['CCDSUM'] == '{0} {1}'.format(*binning), \
        'Incorrect value of CCDSUM recorded'
    assert ad_new[0].hdr['DATASEC'] == '[1:{1},1:{0}]'.format(
        int(1024 / binning[1]), int(1024 / binning[0])), 'Incorrect value for DATASEC ' \
                                                 'recorded'
    assert ad_new[0].hdr['TRIMSEC'] == '[1:{1},1:{0}]'.format(
        int(1024 / binning[1]), int(1024 / binning[0])), 'Incorrect value for TRIMSEC ' \
                                                 'recorded'
    assert ad_new[0].hdr['AMPSIZE'] == '[1:{1},1:{0}]'.format(
        int(1024 / binning[1]), int(1024 / binning[0])), 'Incorrect value for AMPSIZE ' \
                                                 'recorded'
