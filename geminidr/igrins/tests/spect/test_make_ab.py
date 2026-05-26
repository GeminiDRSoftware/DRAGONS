import os
import pytest

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.igrins.primitives_igrins_spect import IGRINS2Spect

INPUT_FILES = (
    ([f"N20260303S{i:04d}_K_ADUToElectrons.fits" for i in range(28, 32)],
     {"processed_arc": "N20260301S0028_K_arc.fits"}),
)

@pytest.fixture
def adinputs(path_to_inputs, request):
    return [astrodata.open(os.path.join(path_to_inputs, f))
                           for f in request.param]


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("adinputs, caldict", INPUT_FILES, indirect=['adinputs'])
def test_make_ab(path_to_inputs, path_to_refs, adinputs, caldict):
    """A simple test for the IGRINS2 makeAB primitive."""
    p = IGRINS2Spect(adinputs)
    # Add the processed arc to the caldb
    p.caldb.user_cals = {k: os.path.join(path_to_inputs, v)
                         for k, v in caldict.items()}
    adout = p.makeAB().pop()
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    # EXTEND is one of those keywords that is put there by the FITS writer
    # so may not be present
    ad_compare(adref, adout, ignore_kw=['EXTEND'])


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("adinputs, caldict", INPUT_FILES, indirect=['adinputs'])
def test_make_ab2(path_to_inputs, path_to_refs, adinputs, caldict):
    """A simple test for the IGRINS2 makeAB primitive."""
    p = IGRINS2Spect(adinputs)
    # Add the processed arc to the caldb
    p.caldb.user_cals = {k: os.path.join(path_to_inputs, v)
                         for k, v in caldict.items()}
    #p.correctFlexure()
    adout = p.makeABNew().pop()
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    # EXTEND is one of those keywords that is put there by the FITS writer
    # so may not be present
    ad_compare(adref, adout, ignore_kw=['EXTEND'])
