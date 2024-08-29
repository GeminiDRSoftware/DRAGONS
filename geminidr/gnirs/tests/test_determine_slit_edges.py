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
    # GNIRS 111/mm LongBlue, off right edge of detector.
    ('N20121118S0375_stack.fits', dict(),
     [[255, 533.6], [511, 530.1], [767, 526.6]]),
    # GNIRS 111/mm LongBlue, off left edge of detector
    ('N20180605S0138_stack.fits', dict(),
     [[255, 483.0], [511, 479.3], [767, 475.6]]),
    # GNIRS 32/mm ShortRed, centered
    ('S20040413S0268_stack.fits', dict(),
     [[255, 504.7], [511, 504.1], [767, 503.4]]),
    # GNIRS 10/mm LongRed, one-off shorter slit length.
    ('N20110718S0129_stack.fits', dict(edges1=10, edges2=906),
     [[255, 454.3], [511, 449.4], [767, 442.9]]),
]

# Cross-dispersed datasets
input_pars_xd = [
    # North Short 32/mm SXD
    ('N20210129S0314_stack.fits', dict(), # 1.65 μm
      [[255, 260.5, 378.5, 455.6, 522.7, 590.3, 663.1],
      [511, 290.8, 400.5, 477.8, 549.3, 624.3, 707.1],
      [767, 319.9, 423.6, 503.3, 581.6, 666.8, 762.9]]),
    # North Short 111 l/mm SXD
    ('N20210131S0096_stack.fits', dict(), # 1.280 μm
      [[255, 289, 399, 476, 547, 624],
      [511, 294, 400, 480, 553, 631],
      [767, 299, 403, 484, 558, 641]]),
    ('N20231030S0026_stack.fits', dict(), # 1.650 μm
      [[255, 277, 389, 467, 539, 610, 688],
      [511, 283, 984, 472, 542, 617, 696],
      [767, 288, 397, 475, 547, 628, 713]]),
    ('N20130708S0154_stack.fits', dict(), # 1.770 μm
      [[255, 264, 385, 464, 527, 594, 664],
      [511, 271, 390, 465, 531, 600, 672],
      [767, 277, 392, 467, 538, 605, 681]]),
    ('N20161105S0385_stack.fits', dict(), # 2.130 μm
      [[255, 266, 384, 462, 527, 596, 667],
      [511, 271, 388, 464, 534, 603, 677],
      [767, 276, 391, 467, 537, 610, 685]]),
    ('N20161105S0407_stack.fits', dict(), # 2.320 μm
      [[255, 304, 415, 493, 561, 640],
      [511, 310, 417, 494, 568, 647],
      [767, 315, 429, 499, 574, 656]]),
    # North Long 10 l/mm SXD
    ('N20170511S0274_stack.fits', dict(), # 1.650 μm
     [[255, 84.2, 437.6, 668.2, 868.1],
      [511, 182.0, 510.3, 741.3, 954.2],
      [767, 277.1, 587.2, 824.6]]),
    # North Long 10 l/mm LXD
    ('N20130821S0308_stack.fits', dict(), # 1.650 μm
      [[255, 165.2, 394.8, 543.0, 670.3, 797.5, 933.5],
      [511, 227.5, 440.2, 588.2, 723.4, 863.9, 1018.4],
      [767, 287.8, 488.1, 639.7, 786.8, 945.2]]),
    ('N20161108S0049_stack.fits', dict(), # 2.2000 μm
      [[255, 185, 406, 552, 687, 821],
      [511, 251, 454, 598, 739],
      [767, 307, 501, 654, 809]]),
    # North Long 32 l/mm SXD
    ('N20101206S0812_stack.fits', dict(), # 2.200 μm
      [[255, 232, 577, 815, 1015],
      [511, 263, 600, 830, 1042],
      [767, 287, 618, 850]]),
    # North Long 32 l/mm LXD
    ('N20201222S0217_stack.fits', dict(), # 1.580 μm
      [[255, 227, 435, 582, 722],
      [511, 246, 451, 600, 741],
      [767, 259, 464, 614, 757]]),
    ('N20201223S0211_stack.fits', dict(), # 1.977 μm
      [[255, 282, 485, 629, 783],
      [511, 298, 495, 650, 800],
      [767, 316, 509, 667]]),
    # North Long 111 l/mm LXD
    ('N20130419S0118_stack.fits', dict(), # 1.942 μm
      [[255, 269, 464, 619, 777],
      [511, 271, 465, 620, 780],
      [767, 273, 468, 623, 783]]),
    ('N20130419S0132_stack.fits', dict(), # 2.002 μm
      [[255, 247, 442, 597, 755],
      [511, 249, 443, 598, 758],
      [767, 251, 444, 601, 761]]),
    ('N20130419S0146_stack.fits', dict(), # 2.062 μm
      [[255, 227, 422, 577, 735],
      [511, 229, 423, 578, 738],
      [767, 231, 424, 581, 741]]),
    ('N20130419S0160_stack.fits', dict(), # 2.122 μm
      [[255, 207, 402, 557, 715],
      [511, 209, 403, 558, 718],
      [767, 211, 404, 561, 721]]),
    ('N20130419S0174_stack.fits', dict(), # 2.182 μm
      [[255, 187, 382, 537, 695],
      [511, 189, 383, 538, 698],
      [767, 191, 384, 541, 701]]),
    ('N20130419S0188_stack.fits', dict(), # 2.242 μm
      [[255, 167, 362, 517, 675],
      [511, 169, 363, 518, 678],
      [767, 171, 364, 521, 681]]),
    ('N20130419S0202_stack.fits', dict(), # 2.302 μm
      [[255, 147, 342, 497, 655],
      [511, 149, 343, 498, 658],
      [767, 151, 344, 501, 661]]),
    # South Short 32 l/mm SXD
    ('S20060507S0138_stack.fits', dict(), # 1.650 μm
      [[255, 292.8, 410.8, 487.8, 554.9, 622.6, 695.7],
      [511, 325.6, 435.1, 512.4, 584.0, 659.2, 742.4],
      [767, 357.2, 460.8, 540.5, 618.9, 704.4, 800.9]]),
    # South Short 111 l/mm SXD
    ('S20060311S0333_stack.fits', dict(), # 1.087 μm
      [[255, 306, 416, 494, 565, 638, 720],
      [511, 314, 521, 501, 574, 649, 734],
      [767, 323, 429, 506, 579, 661]]),
    ('S20041101S0321_stack.fits', dict(), # 1.270 μm
      [[255, 319, 427, 505, 577, 655],
      [511, 326, 436, 512, 586, 667],
      [767, 335, 440, 519, 599, 679]]),
    ('S20060126S0130_stack.fits', dict(), # 1.580 μm
      [[255, 318, 426, 503, 577, 655, 740],
      [511, 326, 431, 511, 587, 666, 756],
      [767, 335, 438, 517, 594, 678]]),
    ('S20060201S0361_stack.fits', dict(), # 1.630 μm
      [[255, 305, 414, 493, 563, 637],
      [511, 318, 423, 501, 571, 647],
      [767, 320, 428, 507, 580, 661]]),
    ('S20051220S0262_stack.fits', dict(), # 1.730 μm
      [[255, 277, 393, 471, 538, 606],
      [511, 285, 399, 476, 543, 613],
      [767, 292, 404, 483, 551, 624]]),
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
