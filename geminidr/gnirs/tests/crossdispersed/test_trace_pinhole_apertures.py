#!/usr/bin/env python3
"""
Tests related to GNIRS Cross-dispersed pinhole mask tracing.
"""
import numpy as np
import os
import pytest

from scipy import ndimage

import astrodata
import geminidr
from astropy.modeling import models
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from gempy.utils import logutils
from recipe_system.testing import ref_ad_factory

datasets = [
    #GNIRS XD pinhole files
    "S20060507S0125_flatCorrected.fits",
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
