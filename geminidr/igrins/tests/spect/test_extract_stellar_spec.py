import os
import pytest

from numpy.testing import assert_allclose

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.igrins.primitives_igrins_spect import IGRINS2Spect


INPUT_FILES = [("N20260303S0028_K_slitProfileEstimated.fits", {"processed_arc": "N20260301S0028_K_arc.fits",
                                                               "processed_flat": "N20260228S0543_K_flat.fits"})]

@pytest.fixture
def ad(path_to_inputs, request):
    return astrodata.open(os.path.join(path_to_inputs, request.param))


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad, caldict", INPUT_FILES, indirect=['ad'])
def test_extract_stellar_spec(path_to_inputs, path_to_refs, change_working_dir, ad, caldict):
    """A simple test for the IGRINS2 extractStellarSpec primitive."""
    p = IGRINS2Spect([ad])
    p.caldb.user_cals = {k: os.path.join(path_to_inputs, v)
                         for k, v in caldict.items()}
    # We want to write this to disk since the wcs will be written then
    p.extractStellarSpec()
    with change_working_dir():
        p.writeOutputs()
        adout = astrodata.open(p.adinputs[0].filename)
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    ad_compare(adref, adout, ignore_kw=[k for k in adref[0].hdr['WAT2*']])
    assert_allclose(adref[0].WAVELENGTHS, adout[0].WAVELENGTHS)
    assert_allclose(adref[0].SN_PER_RESEL, adout[0].SN_PER_RESEL)
