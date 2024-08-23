#!/usr/bin/env python3
"""
Tests for determineSlitEdges() on GNIRS data (longslit and cross-dispersed).
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
    (# GNIRS 111/mm LongBlue, off right edge of detector.
     'N20121118S0375_stack.fits', dict(),
     [[255, 533.6], [511, 530.1], [767, 526.6]]),
    (# GNIRS 111/mm LongBlue, off left edge of detector
     'N20180605S0138_stack.fits', dict(),
     [[255, 483.0], [511, 479.3], [767, 475.6]]),
    (# GNIRS 32/mm ShortRed, centered
     'S20040413S0268_stack.fits', dict(),
     [[255, 504.7], [511, 504.1], [767, 503.4]]),
    (# GNIRS 10/mm LongRed, one-off shorter slit length.
     'N20110718S0129_stack.fits', dict(edges1=10, edges2=906),
     [[255, 454.3], [511, 449.4], [767, 442.9]]),
]

input_pars_xd = [
    (# GNIRS XD 32/mm ShortBlue 1.65 μm
     'N20210129S0314_stack.fits', dict(),
     [[255, 260.5, 378.5, 455.6, 522.7, 590.3, 663.1],
      [511, 290.8, 400.5, 477.8, 549.3, 624.3, 707.1],
      [767, 319.9, 423.6, 503.3, 581.6, 666.8, 762.9]]),
    (# GNIRS XD 10 l/mm LongBlue 1.65 μm
     'N20130821S0308_stack.fits', dict(),
     [[255, 165.2, 394.8, 543.0, 670.3, 797.5, 933.5],
      [511, 227.5, 440.2, 588.2, 723.4, 863.9, 1018.4],
      [767, 287.8, 488.1, 639.7, 786.8, 945.2]]),
    (# GNIRS XD 32 l/mm LongBlue 1.98 μm
     'N20201223S0211_stack.fits', dict(),
     [[255, 282, 485, 629, 783],
      [511, 298, 495, 650, 800],
      [767, 316, 509, 667]]),
    (# GNIRS XD 111 l/mm LongBlue 1.942 μm
     'N20130419S0118_stack.fits', dict(),
     [[255, 269, 464, 619, 777],
      [511, 271, 465, 620, 780],
      [767, 273, 468, 623, 783]]),
    (# GNIRS XD 111 l/mm LongBlue 2.002 μm
     'N20130419S0132_stack.fits', dict(),
     [[255, 247, 442, 597, 755],
      [511, 249, 443, 598, 758],
      [767, 251, 444, 601, 761]]),
    (# GNIRS XD 111 l/mm LongBlue 2.062 μm
     'N20130419S0146_stack.fits', dict(),
     [[255, 227, 422, 577, 735],
      [511, 229, 423, 578, 738],
      [767, 231, 424, 581, 741]]),
    (# GNIRS XD 111 l/mm LongBlue 2.122 μm
     'N20130419S0160_stack.fits', dict(),
     [[255, 207, 402, 557, 715],
      [511, 209, 403, 558, 718],
      [767, 211, 404, 561, 721]]),
    (# GNIRS XD 111 l/mm LongBlue 2.182 μm
     'N20130419S0174_stack.fits', dict(),
     [[255, 187, 382, 537, 695],
      [511, 189, 383, 538, 698],
      [767, 191, 384, 541, 701]]),
    (# GNIRS XD 111 l/mm LongBlue 2.242 μm
     'N20130419S0188_stack.fits', dict(),
     [[255, 167, 362, 517, 675],
      [511, 169, 363, 518, 678],
      [767, 171, 364, 521, 681]]),
    (# GNIRS XD 111 l/mm LongBlue 2.302 μm
     'N20130419S0202_stack.fits', dict(),
     [[255, 147, 342, 497, 655],
      [511, 149, 343, 498, 658],
  [767, 151, 344, 501, 661]]),
]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params,ref_vals", input_pars_ls, indirect=['ad'])
def test_determine_slit_edges_longslit(ad, params, ref_vals):

    p = GNIRSLongslit([ad])
    ad_out = p.determineSlitEdges(**params).pop()

    for midpoints in ref_vals:
        refrow = midpoints.pop(0)
        for i, midpoint in enumerate(midpoints):
            model1 = am.table_to_model(ad_out[0].SLITEDGE[2*i])
            model2 = am.table_to_model(ad_out[0].SLITEDGE[2*i+1])
            print(f"The midpoint at {refrow} is {(model1(refrow) + model2(refrow)) / 2:.1f}")
            assert midpoint == pytest.approx(
                (model1(refrow) + model2(refrow)) / 2, abs=2.)


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params,ref_vals", input_pars_xd, indirect=['ad'])
def test_determine_slit_edges_longslit(ad, params, ref_vals):

    p = GNIRSCrossDispersed([ad])
    ad_out = p.determineSlitEdges(**params).pop()

    for midpoints in ref_vals:
        refrow = midpoints.pop(0)
        for i, midpoint in enumerate(midpoints):
            model1 = am.table_to_model(ad_out[0].SLITEDGE[2*i])
            model2 = am.table_to_model(ad_out[0].SLITEDGE[2*i+1])
            print(f"The midpoint at {refrow} is {(model1(refrow) + model2(refrow)) / 2:.1f}")
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
