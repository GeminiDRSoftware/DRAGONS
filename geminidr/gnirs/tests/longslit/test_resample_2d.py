#!/usr/bin/env python3
"""
Tests for GNIRS LS `resampleToCommonFrame` in 2D.
"""

import os

import numpy as np
import pytest

import astrodata
import gemini_instruments
from geminidr.gnirs.primitives_gnirs_longslit import GNIRSLongslit

# -- Test parameters ----------------------------------------------------------
parameters = {'w1': None, 'w2': None, 'dw': None, 'npix': None,
              'conserve': None, 'interpolant': 'poly3', 'dq_threshold': 0.001}


# -- Datasets -----------------------------------------------------------------
test_datasets = ['N20240329S0022_wcsCorrected.fits', # Central wavelength 1.594 um
                 'N20240329S0023_wcsCorrected.fits',
                 'N20240329S0024_wcsCorrected.fits',
                 'N20240329S0025_wcsCorrected.fits',
                 'N20240329S0040_wcsCorrected.fits', # Central wavelength 1.735 um
                 'N20240329S0041_wcsCorrected.fits',
                 'N20240329S0042_wcsCorrected.fits',
                 'N20240329S0043_wcsCorrected.fits']

# -- Tests --------------------------------------------------------------------
@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
@pytest.mark.parametrize("wavescale", ["linear", "loglinear"])
def test_resample_to_common_frame_with_defaults(input_ad_list, path_to_refs,
                                                change_working_dir, wavescale, caplog):
    p = GNIRSLongslit(input_ad_list)
    p.resampleToCommonFrame(trim_spatial=True, trim_spectral=False,
                            output_wave_scale=wavescale)
    with change_working_dir():
        ad_out = p.stackFrames()[0]
        p.writeOutputs()
        if wavescale == "linear":
            _check_params(caplog.records, 'w1=1525.174 w2=1806.038 dw=0.138 npix=2029')
        else:
            _check_params(caplog.records, 'w1=1525.174 w2=1806.005 dw=0.000077 npix=2197')
        assert 'ALIGN' in ad_out[0].phu
        ref = astrodata.open(os.path.join(
            path_to_refs, f'N20240329S0022_stack_defaults_{wavescale}.fits'))

        np.testing.assert_allclose(ad_out[0].data, ref[0].data)


@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
def test_resample_to_common_frame_trim_spectral(input_ad_list, path_to_refs,
                                                caplog):
    p = GNIRSLongslit(input_ad_list)
    p.resampleToCommonFrame(trim_spatial=True, trim_spectral=True,
                            output_wave_scale="linear")
    ad_out = p.stackFrames()[0]
    _check_params(caplog.records, 'w1=1664.096 w2=1666.589 dw=0.138 npix=19')
    assert 'ALIGN' in ad_out[0].phu
    ref = astrodata.open(os.path.join(path_to_refs,
                                      'N20240329S0022_stack_trim_spectral_True.fits'))

    np.testing.assert_allclose(ad_out[0].data, ref[0].data)


@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
def test_resample_to_common_frame_trim_spatial(input_ad_list, path_to_refs,
                                                caplog):
    p = GNIRSLongslit(input_ad_list)
    p.resampleToCommonFrame(trim_spatial=False, trim_spectral=False,
                            output_wave_scale="linear")
    ad_out = p.stackFrames()[0]
    # This should be the same as test_resample_to_common_frame_with_defaults()
    _check_params(caplog.records, 'w1=1525.174 w2=1806.038 dw=0.138 npix=2029')
    assert 'ALIGN' in ad_out[0].phu
    ref = astrodata.open(os.path.join(path_to_refs,
                                      'N20240329S0022_stack_trim_spatial_False.fits'))

    np.testing.assert_allclose(ad_out[0].data, ref[0].data)


@pytest.mark.gnirsls
@pytest.mark.preprocessed_data
def test_adjust_wavelength_zeropoint_and_resample(input_ad_list):
    """
    Confirms that a zeropoint wavelength shift applied by
    adjustWavelengthZeroPoint() is propagated through the alignment and
    resampling steps.
    """
    p = GNIRSLongslit(input_ad_list[:1])
    orig_wave_limits = p.streams['main'][0][0].wcs(511, (0, 1021))[0]
    p.adjustWavelengthZeroPoint(shift=5)
    wave_limits = p.streams['main'][0][0].wcs(511, (-5, 1016))[0]
    new_wave_limits = p.streams['main'][0][0].wcs(511, (0, 1021))[0]
    np.testing.assert_allclose(wave_limits, orig_wave_limits)
    # This will flip the dispersion to +ve
    p.resampleToCommonFrame()
    resampled_wave_limits = p.streams['main'][0][0].wcs(511, (0, 1021))[0]
    np.testing.assert_allclose(sorted(resampled_wave_limits),
                               sorted(new_wave_limits))


# -- Local fixtures and helper functions --------------------------------------
def _check_params(records, expected):
    for record in records:
        if record.message.startswith('Resampling and linearizing'):
            assert expected in record.message

@pytest.fixture(scope='function')
def input_ad_list(path_to_inputs):

    _input_ad_list = []

    for input_fname in test_datasets:
        input_path = os.path.join(path_to_inputs, input_fname)

        if os.path.exists(input_path):
            ad = astrodata.open(input_path)
        else:
            raise FileNotFoundError(input_path)

        _input_ad_list.append(ad)

    return _input_ad_list
