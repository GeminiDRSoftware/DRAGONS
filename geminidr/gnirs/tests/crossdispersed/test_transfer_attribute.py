#!/usr/bin/env python3
"""
Unit tests for transferAttribute for GNIRS XD
"""
import pytest
from pytest import approx

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed

# -- Datasets -----------------------------------------------------------------
datasets = [("N20210129S0304.fits", # 32 l/mm Short camera
             "N20210129S0323.fits")]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsxd
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("dataset", datasets, indirect=False)
def test_flat_correct(dataset, change_working_dir):
    with change_working_dir():
        adinputs = [astrodata.from_file(download_from_archive(filename)) for
                    filename in dataset]

        p = GNIRSCrossDispersed(adinputs)
        p.prepare()
        p.selectFromInputs(tags='GCAL_IR_OFF,LAMPOFF', outstream='QHLamp')
        p.removeFromInputs(tags='GCAL_IR_OFF,LAMPOFF')
        ad_ref = p.determineSlitEdges(stream='QHLamp', search_radius=60)[0]
        ad_out = p.transferAttribute(stream='main', source='QHLamp',
                                               attribute='SLITEDGE').pop()

        for row1, row2 in zip(ad_out[0].SLITEDGE, ad_ref[0].SLITEDGE):
            for i in range(len(row1)):
                assert row1[i] == approx(row2[i])
