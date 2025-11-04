#!/usr/bin/env python3
"""
Tests for flatCorrect for NIRI LS.
"""
import pytest

import os
import numpy as np

import astrodata, gemini_instruments
from geminidr.niri.primitives_niri_longslit import NIRILongslit
from recipe_system.testing import ref_ad_factory

# -- Datasets -----------------------------------------------------------------
datasets = [("N20050918S0134_nonlinearityCorrected.fits",
             "N20050918S0312_flat.fits"),
            ]

# -- Tests --------------------------------------------------------------------
@pytest.mark.nirils
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, flat", datasets, indirect=True)
def test_flat_correct(ad, flat, ref_ad_factory):
    p = NIRILongslit([ad])
    ad_out = p.flatCorrect(flat=flat).pop()
    ad_ref = ref_ad_factory(ad_out.filename)
    for ext, ext_ref in zip(ad_out, ad_ref):
        np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1e-6)
        np.testing.assert_array_equal(ext.mask, ext_ref.mask)
        np.testing.assert_allclose(ext.variance, ext_ref.variance, rtol=1e-6)


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """Return AD object in input directory"""
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return astrodata.open(path)
    raise FileNotFoundError(path)


@pytest.fixture(scope='function')
def flat(path_to_inputs, request):
    """Return full path to file in input directory"""
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(path)
