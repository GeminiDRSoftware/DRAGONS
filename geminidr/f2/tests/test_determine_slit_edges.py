#!/usr/bin/env python3
"""
Tests for determineSlitEdges() on F2 longslit data.
"""

import numpy as np
import os

import astrodata
import gemini_instruments
from gempy.library import astromodels as am
import pytest

from geminidr.f2.primitives_f2_longslit import F2Longslit

# -- Test Parameters ----------------------------------------------------------

# -- Datasets -----------------------------------------------------------------
input_pars = [
    # (Input file, params, reference values [column: row])
    (# F2 1pix-slit, HK, off left edge of detector.
     'S20140728S0282_stack.fits', dict(),
     [[512, 738.9], [1024, 770.9], [1536, 801.0]]),
    (# F2 2pix-slit, JH.
     'S20131015S0043_stack.fits', dict(spectral_order=4),
     [[512, 770.6], [1024, 769.9], [1536, 772.1]]),
    (# F2 2pix-slit, R3K. Efficiency drops to zero in middle.
     'S20140111S0155_stack.fits', dict(),
     [[512, 778.0], [1024, 773.7], [1536, 771.4]]),
]


# -- Tests --------------------------------------------------------------------
@pytest.mark.f2ls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params,ref_vals", input_pars, indirect=['ad'])
def test_determine_slit_edges_longslit(ad, params, ref_vals):

    # We do this so we don't need to remake the input files if the MDF changes
    del ad.MDF
    p = F2Longslit([ad])
    p.addMDF()
    ad_out = p.determineSlitEdges(**params).pop()

    for midpoints in ref_vals:
        refrow = midpoints.pop(0)
        for i, midpoint in enumerate(midpoints):
            model1 = am.table_to_model(ad_out[0].SLITEDGE[2*i])
            model2 = am.table_to_model(ad_out[0].SLITEDGE[2*i+1])
            assert midpoint == pytest.approx(
                (model1(refrow) + model2(refrow)) / 2, abs=2.)



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
