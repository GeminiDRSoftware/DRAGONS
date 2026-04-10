#!/usr/bin/env python3
"""
Regression tests for F2 spectro darkCorrect.
"""

import os
import numpy as np
import pytest

import astrodata
import gemini_instruments

from geminidr.f2.primitives_f2_longslit import F2Longslit
from recipe_system.testing import ref_ad_factory

test_datasets = [
    ("S20250911S0125_varAdded.fits", "S20250913S0250_dark.fits", ""), # F2 arc with dark
    ("S20250911S0125_varAdded.fits, S20250911S0124_varAdded.fits", "", "_lampoffCorrected"), # F2 arc with lampoff
    ("S20250911S0123_varAdded.fits", "S20250913S0266_dark.fits", ""), # F2 flat with dark
]

@pytest.mark.f2ls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad_files, dark, suffix", test_datasets, indirect=True)
def test_dark_correct(ad_files, dark, suffix, ref_ad_factory):
    p = F2Longslit(ad_files)
    if dark and suffix:
        ad_out = p.darkCorrect(dark=dark, suffix=suffix).pop()
    elif suffix:
        ad_out = p.darkCorrect(suffix=suffix).pop()
    elif dark:
        ad_out = p.darkCorrect(dark=dark).pop()
    else:
        ad_out = p.darkCorrect().pop()
    ad_ref = ref_ad_factory(ad_out.filename)
    for ext, ext_ref in zip(ad_out, ad_ref):
        np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1e-6)
        np.testing.assert_array_equal(ext.mask, ext_ref.mask)
        np.testing.assert_allclose(ext.variance, ext_ref.variance, rtol=1e-6)

# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='function')
def ad_files(path_to_inputs, request):
    """Return AD objects in input directory"""
    filenames = request.param.split(", ")
    for f in filenames:
        path = os.path.join(path_to_inputs, f)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return [astrodata.open(os.path.join(path_to_inputs, f)) for f in filenames]

@pytest.fixture(scope='function')
def dark(path_to_inputs, request):
    """Return full path to file in input directory"""
    if request.param == "":
        return None
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(path)

@pytest.fixture(scope='function')
def suffix(request):
    """Return suffix for output file"""
    return request.param