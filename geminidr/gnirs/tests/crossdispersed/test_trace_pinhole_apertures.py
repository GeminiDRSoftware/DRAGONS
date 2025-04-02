#!/usr/bin/env python3
"""
Tests related to GNIRS Cross-dispersed pinhole mask tracing.
"""
import numpy as np
import os
import pytest

from astropy.stats import sigma_clipped_stats
import astrodata, gemini_instruments
import geminidr
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from recipe_system.testing import ref_ad_factory


datasets = [
    #GNIRS XD pinhole files
    "S20060507S0125_flatCorrected.fits", # 32 l/mm, ShortBlue
    "N20130821S0556_flatCorrected.fits", # 10 l/mm, LongBlue
    "N20231029S0343_stack.fits",         # 111 l/mm, ShortBlue
    ]


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.regression
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_trace_pinhole_apertures(ad, change_working_dir, ref_ad_factory):

    with change_working_dir():
        p = GNIRSCrossDispersed([ad])
        p.viewer = geminidr.dormantViewer(p, None)
        pinholes_traced_ad = p.tracePinholeApertures().pop()

    ref_ad = ref_ad_factory(pinholes_traced_ad.filename)
    for ext, ext_ref in zip(pinholes_traced_ad, ref_ad):
        model = ext.wcs.get_transform('pixels', 'rectified')
        ref_model = ext_ref.wcs.get_transform('pixels', 'rectified')

        assert len(model._parameters) == 8
        assert model.inputs == ('x0', 'x1')
        assert model.outputs == ('z', 'x0')

        X, Y = np.mgrid[:ext.shape[0], :ext.shape[1]]

        np.testing.assert_allclose(model(X, Y), ref_model(X, Y), atol=0.05)


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", datasets, indirect=True)
def test_straight_edges_from_pinhole_model(ad):
    """
    Test that the edges of the masked regions are close to vertical after
    applying the pinhole-created distortion model.
    """
    p = GNIRSCrossDispersed([ad])
    ad = p.tracePinholeApertures().pop()

    for ext in ad:
        nmasked = (ext.mask & 64).astype(bool).sum(axis=1)
        t = ext.wcs.get_transform('pixels', 'rectified')
        y = np.arange(ext.shape[0])
        x1 = ext.mask.argmin(axis=1)
        x2 = ext.shape[1] - ext.mask[:, ::-1].argmin(axis=1) - 1
        # We have to be careful because there are some bad pixels in the inputs
        x1_on = np.logical_and(x1 > 1, nmasked < 0.9 * ext.shape[1])
        x2_on = np.logical_and(x2 < ext.shape[1] - 2, nmasked < 0.9 * ext.shape[1])
        left_transformed = t(x1[x1_on], y[x1_on])
        right_transformed = t(x2[x2_on], y[x2_on])
        assert sigma_clipped_stats(left_transformed[0])[2] < 0.6
        assert sigma_clipped_stats(right_transformed[0])[2] < 0.6

        # Confirm that they round-trip back correctly
        left_round_trip = t.inverse(*left_transformed)
        right_round_trip = t.inverse(*right_transformed)

        # These tolerance seem surprisingly high, but the model is not perfect
        assert np.allclose(left_round_trip[0], x1[x1_on], atol=0.5)
        assert np.allclose(right_round_trip[0], x2[x2_on], atol=0.5)


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
        ad = astrodata.from_file(path)
    else:
        raise FileNotFoundError(path)

    return ad
