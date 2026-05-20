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
def test_save_final_spectra(path_to_inputs, path_to_refs, change_working_dir, ad, caldict):
    """A test for the last few primitives that write output files"""
    p = IGRINS2Spect([ad])
    p.caldb.user_cals = {k: os.path.join(path_to_inputs, v)
                         for k, v in caldict.items()}
    # We want to write this to disk since the wcs will be written then
    with change_working_dir():
        p.extractSpectra()
        p.writeOutputs()
        adout1d = astrodata.open(p.adinputs[0].filename)
        p.saveTwodspec()
        adout2d = astrodata.open(p.adinputs[0].filename.replace("_spec1d", "_spec2d"))
        p.saveDebugImage(save_debug=True)
        adout_debug = astrodata.open(p.adinputs[0].filename.replace("_spec1d", "_spec_debug"))

    # Compare the 1d spectrum _spec1d.fits
    adref1d = astrodata.open(os.path.join(path_to_refs, adout1d.filename))
    ad_compare(adref1d, adout1d, ignore_kw=[k for k in adref1d[0].hdr['WAT2*']])
    assert_allclose(adref1d[0].WAVELENGTHS, adout1d[0].WAVELENGTHS)
    assert_allclose(adref1d[0].SN_PER_RESEL, adout1d[0].SN_PER_RESEL)

    # Compare the 2d spectrum _spec2d.fits
    adref2d = astrodata.open(os.path.join(path_to_refs, adout2d.filename))
    ad_compare(adref2d, adout2d, ignore_kw=[k for k in adref2d[0].hdr['WAT2*']])

    # Compare the debug image _spec_debug.fits
    adref_debug = astrodata.open(os.path.join(path_to_refs, adout_debug.filename))
    ad_compare(adref_debug, adout_debug, ignore_kw=[k for k in adref_debug[0].hdr['WAT2*']])
