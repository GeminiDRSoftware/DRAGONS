#!/usr/bin/env python3
"""
Test related to GNIRS Cross-dispersed spectroscopy arc primitives.

Notes
-----
- The `indirect` argument on `@pytest.mark.parametrize` fixture forces the
  `ad` and `ad_ref` fixtures to be called and the AstroData object returned.
"""
import numpy as np
import os
import pytest

import astrodata, gemini_instruments
import geminidr
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from recipe_system.testing import ref_ad_factory

# Test parameters -------------------------------------------------------------
fixed_parameters_for_determine_distortion = {
    "fwidth": None,
    "id_only": False,
    "max_missed": 1,
    "max_shift": 0.05,
    "min_snr": 5.,
    "nsum": 10,
    "step": 5,
    "spatial_order": 2,
    "spectral_order": 2,
    "min_line_length": 0.8,
    "debug_reject_bad": False
}

input_pars = [
    # Process Arcs: GNIRS
    # (Input File, params)
    # 10 l/mm Longblue SXD
    ('N20170511S0269_wavelengthSolutionDetermined.fits', dict()),
    # 10 l/mm Longblue LXD
    ('N20130821S0301_wavelengthSolutionDetermined.fits', dict()),
    # 32 l/mm Shortblue SXD
    ('N20210129S0324_wavelengthSolutionDetermined.fits', dict()),
    # 111 l/mm Shortblue SXD
    ('N20231030S0034_wavelengthSolutionDetermined.fits', dict()),
    # 32 l/mm Longblue LXD
    ('N20201223S0216_wavelengthSolutionDetermined.fits', dict()),
    # 32 l/mm Shortblue SXD
    ('S20060507S0070_wavelengthSolutionDetermined.fits', dict()),
    # 111 l/mm Shortblue SXD
    ('S20060311S0321_wavelengthSolutionDetermined.fits', dict()),
]

# Tests -----------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params", input_pars, indirect=['ad'])
@pytest.mark.skip("MUST WORK")
def test_regression_for_determine_distortion_using_wcs(
        ad, params, change_working_dir, ref_ad_factory):

    with change_working_dir():
        p = GNIRSCrossDispersed([ad])
        p.determineDistortion(**fixed_parameters_for_determine_distortion)
        distortion_determined_ad = p.writeOutputs().pop()

    ref_ad = ref_ad_factory(distortion_determined_ad.filename)

    errstr = ""
    for ext, ref_ext in zip(distortion_determined_ad, ref_ad):
        # Confirm that the distortion model is placed after the rectification model
        assert (ext.wcs.available_frames.index("distortion_corrected") >
                ext.wcs.available_frames.index("rectified"))
        assert (ref_ext.wcs.available_frames.index("distortion_corrected") >
                ref_ext.wcs.available_frames.index("rectified"))

        model = ext.wcs.get_transform("pixels", "distortion_corrected")
        ref_model = ref_ext.wcs.get_transform("pixels", "distortion_corrected")

        # Otherwise we're doing something wrong!
        assert model[-1].__class__.__name__ == ref_model[-1].__class__.__name__ == "Chebyshev2D"

        Y, X = np.mgrid[:ext.shape[0], :ext.shape[1]]

        # We only care about pixels in the illuminated region
        xx, yy = X[ext.mask == 0], Y[ext.mask == 0]
        diffs = model(xx, yy)[1] - ref_model(xx, yy)[1]  # 1 is y-axis in astropy
        try:
            np.testing.assert_allclose(diffs, 0, atol=1)
        except AssertionError as e:
            errstr += f"Extension {ext.id}\n{str(e)}"

    if errstr:
        raise AssertionError(errstr)


# Local Fixtures and Helper Functions ------------------------------------------
@pytest.fixture(scope='function')
def ad(path_to_inputs, request):
    """
    Returns the pre-processed spectrum file.

    Parameters
    ----------
    path_to_inputs : pytest.fixture
        Fixture defined in :mod:`astrodata.testing` with the path to the
        pre-processed input file.
    request : pytest.fixture
        PyTest built-in fixture containing information about parent test.

    Returns
    -------
    AstroData
        Input spectrum processed up to right before the `distortionDetermine`
        primitive.
    """
    filename = request.param
    path = os.path.join(path_to_inputs, filename)

    if os.path.exists(path):
        ad = astrodata.open(path)
    else:
        raise FileNotFoundError(path)

    return ad
