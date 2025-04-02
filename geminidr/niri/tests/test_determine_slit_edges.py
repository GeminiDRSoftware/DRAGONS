#!/usr/bin/env python3
"""
Tests for determineSlitEdges() on NIRI longslit data.
"""

import numpy as np
import os

import astrodata
import gemini_instruments
from gempy.library import astromodels as am
import pytest

from geminidr.niri.primitives_niri_longslit import NIRILongslit

# -- Test Parameters ----------------------------------------------------------

# -- Datasets -----------------------------------------------------------------
input_pars = [
    # (Input file, params, reference values [column: row])
    (# NIRI f/6 4pix "blue" slit
     'N20081223S0263_stack.fits', dict(),
     ([256, 494.0], [512, 495.2], [767, 496.8])),
    (# NIRI f/32 10pix slit, which is also the f/6 2pix slit
     'N20090925S0312_stack.fits', dict(),
     ([256, 505.1], [512, 506.0], [767, 506.8])),
]


# -- Tests --------------------------------------------------------------------
@pytest.mark.nirils
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params,ref_vals", input_pars, indirect=['ad'])
def test_determine_slit_edges_longslit(ad, params, ref_vals):

    # We do this so we don't need to remake the input files if the MDF changes
    del ad.MDF
    p = NIRILongslit([ad])
    p.addMDF()
    ad_out = p.determineSlitEdges(**params).pop()

    for midpoints in ref_vals:
        refcol = midpoints.pop(0)
        for i, midpoint in enumerate(midpoints):
            model1 = am.table_to_model(ad_out[0].SLITEDGE[2*i])
            model2 = am.table_to_model(ad_out[0].SLITEDGE[2*i+1])
            print(f"The midpoint at {refcol} is {(model1(refcol) + model2(refcol)) / 2:.1f}")
            assert midpoint == pytest.approx(
                (model1(refcol) + model2(refcol)) / 2, abs=2.)



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
