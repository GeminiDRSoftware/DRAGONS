import pytest

import os
import numpy as np

import astrodata, gemini_instruments
from geminidr.gmos.primitives_gmos_longslit import GMOSLongslit
from recipe_system.testing import ref_ad_factory

datasets = [("N20091018S0013_wavelengthSolutionAttached.fits", "N20090920S0020_flat.fits"),
            ("N20180908S0020_wavelengthSolutionAttached.fits", "N20180908S0019_flat.fits")]

@pytest.mark.gmosls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, flat", datasets, indirect=True)
def test_flat_correct(ad, flat, ref_ad_factory):
    p = GMOSLongslit([ad])
    ad_out = p.flatCorrect(flat=flat).pop()
    ad_ref = ref_ad_factory(ad_out.filename)
    for ext, ext_ref in zip(ad_out, ad_ref):
        np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1e-6)
        np.testing.assert_array_equal(ext.mask, ext_ref.mask)
        np.testing.assert_allclose(ext.variance, ext_ref.variance, rtol=1e-6)


@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """Return AD object in input directory"""
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return astrodata.from_file(path)
    raise FileNotFoundError(path)


@pytest.fixture(scope='function')
def flat(path_to_inputs, request):
    """Return full path to file in input directory"""
    path = os.path.join(path_to_inputs, request.param)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(path)
