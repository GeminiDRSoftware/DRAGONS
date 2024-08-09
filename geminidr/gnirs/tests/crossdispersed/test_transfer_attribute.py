#!/usr/bin/env python3
"""
Unit tests for transferAttribute for GNIRS XD
"""
import pytest
from pytest import approx

import os
import numpy as np

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from recipe_system.testing import ref_ad_factory

# -- Datasets -----------------------------------------------------------------
datasets = [("N20210129S0304_varAdded.fits", # 32 l/mm Short camera
             "N20210129S0323_varAdded.fits")]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("dataset", datasets, indirect=False)
def test_flat_correct(dataset, path_to_inputs):#, ref_ad_factory):
    adinputs = [astrodata.open(os.path.join(path_to_inputs, filename)) for
                filename in dataset]

    p = GNIRSCrossDispersed(adinputs)
    p.selectFromInputs(tags='GCAL_IR_OFF,LAMPOFF', outstream='QHLamp')
    p.removeFromInputs(tags='GCAL_IR_OFF,LAMPOFF')
    ad_ref = p.determineSlitEdges(stream='QHLamp', search_radius=60)[0]
    ad_out = p.transferAttribute(stream='main', source='QHLamp',
                                           attribute='SLITEDGE').pop()

    for row1, row2 in zip(ad_out[0].SLITEDGE, ad_ref[0].SLITEDGE):
        for i in range(len(row1)):
            assert row1[i] == approx(row2[i])
