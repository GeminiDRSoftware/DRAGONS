import os
import pytest

from numpy.testing import assert_allclose

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.igrins.primitives_igrins_spect import IGRINS2Spect


INPUT_FILES = [("N20260303S0028_K_AB.fits", {"processed_arc": "N20260301S0028_K_arc.fits",
                                             "processed_flat": "N20260228S0543_K_flat.fits"})]

@pytest.fixture
def ad(path_to_inputs, request):
    return astrodata.open(os.path.join(path_to_inputs, request.param))


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, caldict", INPUT_FILES, indirect=['ad'])
def test_estimate_slit_profile(path_to_inputs, path_to_refs, ad, caldict):
    """A simple test for the IGRINS2 estimateSlitProfile primitive."""
    p = IGRINS2Spect([ad])
    p.caldb.user_cals = {k: os.path.join(path_to_inputs, v)
                         for k, v in caldict.items()}
    adout = p.estimateSlitProfile().pop()
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    ad_compare(adref, adout)
    assert_allclose(adref[0].SLITPROFILE_MAP, adout[0].SLITPROFILE_MAP)
