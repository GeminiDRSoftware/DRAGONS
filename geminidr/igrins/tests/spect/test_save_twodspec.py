import os
import pytest

from numpy.testing import assert_allclose

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.igrins.primitives_igrins_spect import IGRINS2Spect


# The input file had to be manually extracted from the debug
# stream at the end of extractStellarSpect
INPUT_FILES = [("N20260303S0028_K_intermediate.fits", {"processed_arc": "N20260301S0028_K_arc.fits"})]

@pytest.fixture
def ad(path_to_inputs, request):
    return astrodata.open(os.path.join(path_to_inputs, request.param))


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, caldict", INPUT_FILES, indirect=['ad'])
def test_save_twodspec(path_to_inputs, path_to_refs, change_working_dir, ad, caldict):
    """A simple test for the IGRINS2 saveTwodSpec primitive."""
    p = IGRINS2Spect([])
    p.streams["debug"] = [ad]
    p.caldb.user_cals = {k: os.path.join(path_to_inputs, v)
                         for k, v in caldict.items()}
    # We want to write this to disk since the wcs will be written then
    with change_working_dir():
        p.saveTwodspec()
        adout = astrodata.open(ad.filename.replace("_intermediate", "_spec2d"))
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    ad_compare(adref, adout, ignore_kw=[k for k in adref[0].hdr['WAT2*']])
    assert_allclose(adref[0].WAVELENGTHS, adout[0].WAVELENGTHS)
