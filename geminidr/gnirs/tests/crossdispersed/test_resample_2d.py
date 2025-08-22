"""
Regression tests for GNIRS XD `resampleToCommonFrame`.
"""
import os

import pytest

import numpy as np

import astrodata, gemini_instruments
from geminidr.gnirs.primitives_gnirs_crossdispersed import GNIRSCrossDispersed
from gempy.utils import logutils

# Test parameters ------------------------------------------------------
test_datasets = (
    # Standard star observations, ABBA sequence, same wavelengths
    "S20060507S0048_distortionCorrected.fits",
    "S20060507S0049_distortionCorrected.fits",
    "S20060507S0050_distortionCorrected.fits",
    "S20060507S0051_distortionCorrected.fits",
)

# Widths of the extensions after cutting
ext_widths = (157, 130, 135, 159, 197, 245)

# Local fixtures and helper functions ----------------------------------
@pytest.fixture(scope='function')
def adinputs(path_to_inputs):
    return [astrodata.open(os.path.join(path_to_inputs, f))
            for f in test_datasets]

@pytest.fixture(scope="module", autouse=True)
def setup_log(change_working_dir):
    with change_working_dir():
        logutils.config(file_name='test_gnirs_spect_xd_resample_2d.log')

# Tests ----------------------------------------------------------------

@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_resampling_single_wave_scale(adinputs, caplog):
    p = GNIRSCrossDispersed(adinputs)
    p.adjustWCSToReference()
    # default is to linearize
    adoutputs = p.resampleToCommonFrame(single_wave_scale=True)
    wpars = 'w1=703.309 w2=2519.200 dw=0.237 npix=7660'
    assert any(wpars in record.message for record in caplog.records)
    for ad in adoutputs:
        for i, ext in enumerate(ad):
            assert ext.shape == (7660, ext_widths[i])


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_resampling_and_w1_w2(adinputs, caplog):
    p = GNIRSCrossDispersed(adinputs)
    p.adjustWCSToReference()
    # default is to linearize
    adoutputs = p.resampleToCommonFrame(w1=700.000, w2=2520.160, dw=0.237)
    wpars = 'w1=700.000 w2=2520.160 dw=0.237 npix=7681'
    assert any(wpars in record.message for record in caplog.records)
    for ad in adoutputs:
        for ext in ad:
            w = ext.wcs(0, np.arange(ext.shape[0]))[0]
            np.testing.assert_allclose((w.min(), w.max()), (700, 2520.16))
            # Check it's been linearized
            diffs = np.diff(w)
            np.testing.assert_allclose(diffs, diffs[0])


@pytest.mark.gnirsxd
@pytest.mark.preprocessed_data
def test_resampling_separate_orders(adinputs):
    p = GNIRSCrossDispersed(adinputs)
    waves = [ext.wcs(0, (0, 1021))[0] for ext in adinputs[0]]
    p.adjustWCSToReference()
    # default is to linearize
    adoutputs = p.resampleToCommonFrame(single_wave_scale=False)
    for ad in adoutputs:
        for ext, ext_waves in zip(ad, waves):
            w = ext.wcs(0, np.arange(ext.shape[0]))[0]
            np.testing.assert_allclose((w.min(), w.max()), sorted(ext_waves))
            # Check it's been linearized
            diffs = np.diff(w)
            np.testing.assert_allclose(diffs, diffs[0])
            # Check number of dispersion pixels has been preserved
            assert ext.shape[0] == 1022
