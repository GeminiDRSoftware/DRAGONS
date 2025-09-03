import os
import pytest

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed


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
