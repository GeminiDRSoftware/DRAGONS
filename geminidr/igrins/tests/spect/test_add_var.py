import os
import pytest

import astrodata, gemini_instruments
from astrodata.testing import ad_compare

from geminidr.igrins.primitives_igrins_spect import IGRINS2Spect


INPUT_FILES = ["N20260303S0028_K_dqAdded.fits"]

@pytest.fixture
def ad(path_to_inputs, request):
    return astrodata.open(os.path.join(path_to_inputs, request.param))


@pytest.mark.igrins2
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("ad", INPUT_FILES, indirect=True)
def test_add_var(path_to_refs, ad):
    """A simple test for the IGRINS2 addVAR primitive."""
    p = IGRINS2Spect([ad])
    adout = p.addVAR(read_noise=True, poisson_noise=True).pop()
    adref = astrodata.open(os.path.join(path_to_refs, adout.filename))
    ad_compare(adref, adout)
