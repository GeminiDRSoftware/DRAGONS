#!/usr/bin/env python3
"""
Tests for flatCorrect for F2 LS.
"""
import pytest

import os
import numpy as np

import astrodata, gemini_instruments
from geminidr.f2.primitives_f2_longslit import F2Longslit
from recipe_system.testing import ref_ad_factory

# -- Datasets -----------------------------------------------------------------
datasets = [("S20150629S0230_darkCorrected.fits",
             "S20150724S0216_flat.fits"),
            ("S20170215S0111_darkCorrected.fits",
              "S20170211S0230_flat.fits")
            ]

# -- Tests --------------------------------------------------------------------
@pytest.mark.f2ls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, flat", datasets, indirect=True)
def test_flat_correct(ad, flat, ref_ad_factory):
    p = F2Longslit([ad])
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
