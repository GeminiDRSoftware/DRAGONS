#!/usr/bin/env python3
"""
Tests for determineSlitEdges() on GNIRS XD data
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

# Cross-dispersed datasets
input_pars_xd = [
    # North Short 32/mm SXD
    ('N20210129S0314_stack.fits', dict(), # 1.65 μm
     {255: (260.5, 378.5, 455.6, 522.7, 590.3, 663.1),
      511: (290.8, 400.5, 477.8, 549.3, 624.3, 707.1),
      767: (319.9, 423.6, 503.3, 581.6, 666.8, 762.9)}),
    # North Short 111 l/mm SXD
    ('N20210131S0096_stack.fits', dict(), # 1.280 μm
     {255: (287.4, 396.8, 474.3, 546.3, 621.9),
      511: (293.2, 400.7, 478.6, 552.3, 630.5),
      767: (298.7, 404.5, 483.2, 558.8, 641.1)}),
    ('N20231030S0026_stack.fits', dict(), # 1.650 μm
     {255: (277.2, 389.9, 467.0, 536.8, 609.0, 688.2),
      511: (283.1, 393.5, 470.8, 542.1, 616.6, 698.8),
      767: (288.6, 397.1, 474.9, 547.8, 625.0, 710.7)}),
    ('N20130808S0260_stack.fits', dict(), # 1.770 μm
     {255: (262.1, 383.6, 461.1, 527.2, 593.0, 663.3),
      511: (268.4, 387.0, 464.2, 531.2, 598.7, 671.3),
      767: (274.2, 390.3, 467.4, 535.6, 605.0, 680.3)}),
    ('N20161105S0385_stack.fits', dict(), # 2.130 μm
     {255: (266, 384, 462, 527, 596, 667),
      511: (271, 388, 464, 534, 603, 677),
      767: (276, 391, 467, 537, 610, 685)}),
    ('N20161105S0407_stack.fits', dict(), # 2.320 μm
     {255: (303.4, 412.7, 490.2, 561.4, 638.3),
      511: (309.2, 416.5, 494.5, 568.3, 646.9),
      767: (314.7, 420.4, 499.1, 575.0, 656.6)}),
    # North Long 10 l/mm SXD
    ('N20170511S0274_stack.fits', dict(), # 1.650 μm
     {255: (84.2, 437.6, 668.2, 868.1),
      511: (182.0, 510.3, 741.3, 954.2),
      767: (277.1, 587.2, 824.6)}),
    # North Long 10 l/mm LXD
    ('N20130821S0308_stack.fits', dict(), # 1.650 μm
     {255: (165.2, 394.8, 543.0, 670.3, 797.5, 933.5),
      511: (227.5, 440.2, 588.2, 723.4, 863.9, 1018.4),
      767: (287.8, 488.1, 639.7, 786.8, 945.2)}),
    ('N20161108S0049_stack.fits', dict(), # 2.2000 μm
     {255: (184.7, 403.3, 550.9, 682.6, 817.4, 966),
      511: (245.6, 450.0, 599.6, 741.7, 894),
      767: (305.2, 500.2, 656.2, 813.1)}),
    # North Long 32 l/mm SXD
    ('N20101206S0812_stack.fits', dict(), # 2.200 μm
     {255: (232, 577, 810, 1015),
      511: (263, 600, 830, 1042),
      767: (287, 618, 850)}),
    # North Long 32 l/mm LXD
    ('N20201222S0217_stack.fits', dict(), # [10] 1.580 μm
     {255: (227, 435, 582, 722, 870),
      511: (246, 451, 600, 741, 896),
      767: (259, 464, 614, 760, 924)}),
    ('N20201223S0211_stack.fits', dict(), # 1.977 μm
     {255: (282, 485, 629, 783),
      511: (298, 495, 650, 800),
      767: (316, 509, 667, 830)}),
    # North Long 111 l/mm LXD
    ('N20130419S0118_stack.fits', dict(), # 1.942 μm
     {255: (269, 464, 619, 777),
      511: (271, 465, 620, 780),
      767: (273, 468, 623, 783)}),
    ('N20130419S0132_stack.fits', dict(), # 2.002 μm
     {255: (247, 442, 597, 747),
      511: (249, 443, 598, 750),
      767: (251, 444, 601, 754)}),
    ('N20130419S0146_stack.fits', dict(), # 2.062 μm
     {255: (222, 422, 577, 720, 876),
      511: (225, 423, 578, 725, 880),
      767: (228, 424, 581, 729, 884)}),
    ('N20130419S0160_stack.fits', dict(), # [15] 2.122 μm
     {255: (200, 406, 557, 698, 846),
      511: (202, 408, 558, 701, 850),
      767: (204, 410, 561, 704, 854)}),
    ('N20130419S0174_stack.fits', dict(), # 2.182 μm
     {255: (179, 389, 537, 675, 816),
      511: (181, 390, 538, 678, 821),
      767: (183, 391, 541, 681, 826)}),
    ('N20130419S0188_stack.fits', dict(), # 2.242 μm
     {255: (152, 370, 517, 654, 791),
      511: (156, 371, 518, 656, 795),
      767: (160, 372, 521, 658, 799)}),
    ('N20130419S0202_stack.fits', dict(), # 2.302 μm
     {255: (133, 357, 504, 633, 765),
      511: (136, 359, 505, 634, 768),
      767: (139, 361, 506, 635, 771)}),
    # South Short 32 l/mm SXD
    ('S20060507S0138_stack.fits', dict(), # 1.650 μm
     {255: (292.8, 410.8, 487.8, 554.9, 622.6, 695.7),
      511: (325.6, 435.1, 512.4, 584.0, 659.2, 742.4),
      767: (357.2, 460.8, 540.5, 618.9, 704.4, 800.9)}),
    # South Short 111 l/mm SXD
    ('S20060311S0333_stack.fits', dict(), # [20] 1.087 μm
     {255: (306, 416, 494, 565, 638, 720),
      511: (314, 421, 501, 574, 649, 734),
      767: (323, 429, 506, 579, 661)}),
    ('S20041101S0321_stack.fits', dict(), # 1.270 μm
     {255: (319.2, 427.1, 504.7, 577.5, 655.1),
      511: (327.6, 433.8, 511.8, 586.6, 666.9),
      767: (335.9, 440.6, 519.4, 596.3, 679.8)}),
    ('S20060126S0130_stack.fits', dict(), # 1.580 μm
     {255: (318, 426, 503, 577, 655, 740),
      511: (326, 431, 511, 587, 666, 756),
      767: (335, 438, 517, 594, 678)}),
    ('S20060201S0361_stack.fits', dict(), # 1.630 μm
     {255: (305, 414, 493, 563, 637, 720),
      511: (313, 423, 501, 571, 647, 735),
      767: (320, 428, 507, 580, 661, 750)}),
    ('S20051220S0262_stack.fits', dict(), # 1.730 μm
     {255: (277, 393, 471, 538, 606, 680),
      511: (285, 399, 476, 543, 613, 692),
      767: (292, 404, 483, 551, 624, 704)}),
    ('N20130630S0177_stack.fits', dict(), # faint order 7
     {255: (243, 442, 589, 745, 907),
      511: (245, 443, 595, 748, 914),
      767: (247, 445, 599, 752, 921)}),
    ('N20200818S0137_stack.fits', dict(), # faint order 8
     {255: (125, 345, 492, 624, 760, 909),
      511: (128, 346, 493, 625, 764, 914),
      767: (130, 347, 495, 627, 767, 920)}),
]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad,params,ref_vals", input_pars_xd, indirect=['ad'])
def test_determine_slit_edges_crossdispersed(ad, params, ref_vals):

    # We do this so we don't need to remake the input files if the MDF changes
    del ad.MDF
    p = GNIRSCrossDispersed([ad])
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
