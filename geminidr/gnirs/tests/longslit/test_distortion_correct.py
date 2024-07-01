#!/usr/bin/env python3
"""
Tests for distortionCorrect() for GNIRS data.
"""

from pathlib import Path

import numpy as np
import pytest

import astrodata
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

# -- Datasets -----------------------------------------------------------------

test_files = ['N20220706S0306_readoutCleaned.fits', # LongBlue
              'N20170601S0291_readoutCleaned.fits', # ShortBlue
              'N20180114S0121_readoutCleaned.fits', # LongRed
              'N20111231S0352_readoutCleaned.fits'] # ShortRed

# -- Tests --------------------------------------------------------------------

@pytest.mark.parametrize("filename", test_files)
def test_distortion_correct(filename, path_to_inputs, path_to_refs,
                            change_working_dir):

    parameters = {'spatial_order': 3, 'spectral_order': 4, 'id_only': False,
                  'min_snr': 5., 'fwidth': None, 'nsum': 10, 'step': 10,
                  'max_shift': 0.05, 'max_missed': 5, 'min_line_length': 0.}

    with change_working_dir(path_to_inputs):
        ad_in = astrodata.open(filename)

    with change_working_dir(path_to_refs):
        ad_ref = astrodata.open(filename.replace('_readoutCleaned.fits',
                                                 '_distortionCorrected.fits'))

    p = GNIRSLongslit([ad_in])
    ad_out = p.distortionCorrect()[0]

    for ext_out, ext_ref in zip(ad_out, ad_ref):
        np.testing.assert_allclose(ext_out.data, ext_ref.data)
