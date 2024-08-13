#!/usr/bin/env python3
"""
Tests for `cutSlits` on GNIRS cross-dispersed data.
"""
import os

import pytest
from pytest import approx

import astrodata
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed

# -- Test datasets ------------------------------------------------------------
files = (
    'N20210129S0304_attributeTransferred.fits', # 32 l/mm ShortBlue
    'S20060507S0128_attributeTransferred.fits', # 32 l/mm ShortBlue
    'N20130821S0302_attributeTransferred.fits', # 10 l/mm LongBlue
    )

# -- Tests --------------------------------------------------------------------
@pytest.mark.skip("Needs fix for saving and reconstructing WCS giving different structure")
@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize('adinputs', files)
def test_cut_slits(adinputs, path_to_inputs):
    """
    Check that, upon the slits being cut out, input coordinates are recovered
    successfully when transformed to world coordinates and back.
    """
    p = GNIRSCrossDispersed([astrodata.open(os.path.join(path_to_inputs, adinputs))])
    adout = p.cutSlits()[0]

    for ext in adout:
        mid_x = ext.shape[1] / 2.
        mid_y = ext.shape[0] / 2.
        # Check the midpoint of the extension
        assert ext.wcs.backward_transform(
            *ext.wcs.forward_transform(mid_x, mid_y)) == approx((mid_x, mid_y),
                                                                rel=1e-3)
        # Check the corners (need 'abs' for this since we're comparing to zero)
        for i in (0, ext.shape[0]-1):
            for j in (0, ext.shape[1]-1):
                assert ext.wcs.backward_transform(
                    *ext.wcs.forward_transform(i, j)) == approx((i, j),
                                                                rel=1e-3,
                                                                abs=1e-8)
