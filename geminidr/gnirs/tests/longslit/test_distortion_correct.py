#!/usr/bin/env python3
"""
Tests for distortionCorrect() for GNIRS data.
"""

from pathlib import Path

import numpy as np
import pytest
from pytest import approx

import astrodata
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

# -- Datasets -----------------------------------------------------------------

test_files = ['N20220706S0306_readoutCleaned.fits', # LongBlue
              'N20170601S0291_readoutCleaned.fits', # ShortBlue
              'N20180114S0121_readoutCleaned.fits', # LongRed
              'N20111231S0352_readoutCleaned.fits'] # ShortRed

# -- Test parameters ----------------------------------------------------------
parameters = {'interpolant': 'poly3', 'subsample': 1, 'dq_threshold': 0.001}

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("filename", test_files)
def test_distortion_correct(filename, path_to_inputs, path_to_refs,
                            change_working_dir):

    with change_working_dir(path_to_inputs):
        ad_in = astrodata.from_file(filename)

    with change_working_dir(path_to_refs):
        ad_ref = astrodata.from_file(filename.replace('_readoutCleaned.fits',
                                                 '_distortionCorrected.fits'))

    p = GNIRSLongslit([ad_in])
    ad_out = p.distortionCorrect(**parameters)[0]

    for ext_out, ext_ref in zip(ad_out, ad_ref):
        np.testing.assert_allclose(ext_out.data, ext_ref.data)

        # Check that pixel coordinates remain (roughly) the same after being
        # converted to world coordinates and back.
        mid_x = ext_out.shape[1] / 2.
        mid_y = ext_out.shape[0] / 2.
        assert ext_out.wcs.backward_transform(
            *ext_out.wcs.forward_transform(mid_x, mid_y)) == approx((mid_x, mid_y),
                                                                    rel=1e-3)
