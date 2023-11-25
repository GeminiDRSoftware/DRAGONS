"""
Regression tests for GNIRS XD `resampleToCommonFrame`.
"""
import os

import pytest

import astrodata
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_crosssdispersed import GNIRSCrossDispersed
from gempy.utils import logutils

# Test paremeters ------------------------------------------------------
test_datasets = (
    # Standard star observations
    "S20060507S0048_distortionCorrected.fits",
    "S20060507S0049_distortionCorrected.fits",
    "S20060507S0050_distortionCorrected.fits",
    "S20060507S0051_distortionCorrected.fits",
)

# Widths of the
ext_widths = (157, 130, 135, 159, 197, 245)

# Local fixtures and helper functions ----------------------------------
@pytest.fixture(scope='function')
def adinputs(path_to_inputs):
    return [astrodata.open(os.path.join(path_to_inputs, f))
            for f in test_datasets]

def _check_params(records, expected):
    assert len(records) > 0  # make sure caplog is capturing something!
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message

@pytest.fixture(scope="module", autouse=True)
def setup_log(change_working_dir):
    with change_working_dir():
        logutils.config(file_name='test_gnirs_spect_xd_resample_2d.log')

# Tests ----------------------------------------------------------------

@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_resampling(adinputs, caplog):

    p = GNIRSCrossDispersed(adinputs)
    p.adjustWCSToReference()
    adout = p.resampleToCommonFrame()
    _check_params(caplog.records, 'w1=703.092 w2=2519.096 dw=0.237 npix=7654')
    for ad in adout:
        for i, ext in enumerate(ad):
            assert ext.shape == (7654, ext_widths[i])

@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_resampling_and_w1_w2(adinputs, caplog):
    p = GNIRSCrossDispersed(adinputs)
    p.adjustWCSToReference()
    adout = p.resampleToCommonFrame(w1=700.000, w2=2520.160, dw=0.237)
    _check_params(caplog.records, 'w1=700.000 w2=2520.160 dw=0.237 npix=7681')
