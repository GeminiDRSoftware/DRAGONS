import os
import pytest

import numpy as np

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from geminidr.gemini.lookups import DQ_definitions as DQ

@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_combine_orders(path_to_inputs):
    """
    Test to be developed
    """
    ad = astrodata.open(os.path.join(path_to_inputs, "N20220816S0494_1D.fits"))
    napertures = len(set(ad.hdr['APERTURE']))

    p = GNIRSCrossDispersed([ad])

    adout = p.combineOrders().pop()

    assert len(adout) == napertures

@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_mark_beyond_regions(path_to_inputs):
    """
    Test the GNIRS XD markByondRegions primitive.
    """
    ad = astrodata.open(os.path.join(path_to_inputs, "N20220816S0494_1D.fits"))

    p = GNIRSCrossDispersed([ad])

    adout = p.maskBeyondRegions(regions5="1180:1200,1210:", aperture=2).pop()

    assert np.all(adout[5].mask[:141] & DQ.no_data)  # all masked before 1180
    assert np.all(adout[5].mask[193:219] & DQ.no_data)
    assert not np.any(adout[5].mask[219:1000] & DQ.no_data)