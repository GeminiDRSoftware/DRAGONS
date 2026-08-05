#!/usr/bin/env python3
"""
Tests for determineSlitEdges() on GNIRS longslit data
"""

import numpy as np
import os

import astrodata
import gemini_instruments
from gempy.library import astromodels as am
import pytest

from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed

# -- Test Parameters ----------------------------------------------------------

# -- Datasets -----------------------------------------------------------------
input_pars_ls = [
    # (Input file, params, reference values [column: row])
    # GNIRS 111/mm LongBlue, off right edge of detector.
    ('N20121118S0375_stack.fits', dict(),
     {255: (530,), 511: (526,), 767: (522,)}),
    # GNIRS 111/mm LongBlue, off left edge of detector
    ('N20180605S0138_stack.fits', dict(),
     {255: (490,), 511: (486,), 767: (482,)}),
    # GNIRS 32/mm ShortRed, centered
    ('S20040413S0268_stack.fits', dict(),
     {255: (504.7,), 511: (504.1,), 767: (503.4,)}),
    # GNIRS 10/mm LongRed, one-off shorter slit length.
    ('N20110718S0129_stack.fits', dict(),
     {255: (454.3,), 511: (449.4,), 767: (442.9,)}),
]


# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params,ref_vals", input_pars_ls, indirect=['ad'])
def test_determine_slit_edges_longslit(ad, params, ref_vals):

    # We do this so we don't need to remake the input files if the MDF changes
    del ad.MDF
    p = GNIRSLongslit([ad])
    p.addMDF()
    ad_out = p.determineSlitEdges(**params).pop()

    for refrow, midpoints in ref_vals.items():
        for i, midpoint in enumerate(midpoints):
            model1 = am.table_to_model(ad_out[0].SLITEDGE[2*i])
            model2 = am.table_to_model(ad_out[0].SLITEDGE[2*i+1])
            assert midpoint == pytest.approx(
                (model1(refrow) + model2(refrow)) / 2, abs=5.)


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
