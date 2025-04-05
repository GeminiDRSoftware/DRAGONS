#!/usr/bin/env python3
"""
Tests for distortionCorrect() for GNIRS cross-dispersed data.
"""

import os

# from astropy import units as u
import pytest
from pytest import approx

import astrodata
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed

# -- Datasets -----------------------------------------------------------------
test_files = (
    "N20130821S0322_pinholeModelAttached.fits", # 10 l/mm LongBlue
    "N20210129S0296_pinholeModelAttached.fits", # 32 l/mm ShortBlue
    "S20060507S0062_pinholeModelAttached.fits", # 32 l/mm ShortBlue
    "N20210131S0084_pinholeModelAttached.fits", # 111 l/mm ShortBlue
    )

# -- Test parameters ----------------------------------------------------------
params_distcorr = {'interpolant': 'poly3', 'subsample': 1, 'dq_threshold': 0.001}

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("filename", test_files)
def test_distortion_correct_coords_roundtrip(filename, path_to_inputs):

    ad_in = astrodata.from_file(os.path.join(path_to_inputs, filename))

    abs_diff = 10 if 'Long' in ad_in.camera() else 6 # Roughly 1" for both

    for ext in ad_in:
        mid_x = ext.shape[1] / 2.
        mid_y = ext.shape[0] / 2.
        # Check the midpoint of the extension
        assert ext.wcs.backward_transform(
            *ext.wcs.forward_transform(mid_x, mid_y)) == approx((mid_x, mid_y),
                                                                rel=1e-2)
        # Check the corners (need 'abs' for this since we're comparing to zero)
        for i in (0, ext.shape[0]-1):
            for j in (0, ext.shape[1]-1):
                assert ext.wcs.backward_transform(
                    *ext.wcs.forward_transform(i, j)) == approx(
                        (i, j), rel=1e-2, abs=abs_diff)

    p = GNIRSCrossDispersed([ad_in])
    ad_out = p.distortionCorrect(**params_distcorr)[0]

    for ext in ad_out:
        mid_x = ext.shape[1] / 2.
        mid_y = ext.shape[0] / 2.
        # Check the midpoint of the extension
        assert ext.wcs.backward_transform(
            *ext.wcs.forward_transform(mid_x, mid_y)) == approx((mid_x, mid_y),
                                                                rel=1e-2)
        # Check the corners (need 'abs' for this since we're comparing to zero)
        for i in (0, ext.shape[0]-1):
            for j in (0, ext.shape[1]-1):
                assert ext.wcs.backward_transform(
                    *ext.wcs.forward_transform(i, j)) == approx(
                        (i, j), rel=1e-2, abs=abs_diff)
