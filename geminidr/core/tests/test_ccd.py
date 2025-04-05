"""
Tests applied to primitives_ccd.py
"""

import numpy as np
import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos_image import GMOSImage

datasets = ["N20190101S0001.fits"]  # 4x4 binned so limit is defo 65535


# -- Tests --------------------------------------------------------------------


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("raw_ad", datasets, indirect=True)
def test_saturation_level_modification_in_overscan_correct(raw_ad):
    """Confirm that the saturation_level descriptor return is modified
    when the bias level is subtracted by overscanCorrect()"""
    p = GMOSImage([raw_ad])  # modify if other instruments are used as well
    assert raw_ad.saturation_level() == [65535] * len(raw_ad)
    p.prepare()
    assert raw_ad.saturation_level() == [65535] * len(raw_ad)
    p.overscanCorrect()
    bias_levels = np.asarray(raw_ad.hdr['OVERSCAN'])
    np.testing.assert_allclose(raw_ad.saturation_level(), 65535 - bias_levels)
    np.testing.assert_allclose(raw_ad.saturation_level(), raw_ad.non_linear_level())


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='function')
def raw_ad(request):
    filename = request.param
    raw_ad = astrodata.from_file(download_from_archive(filename))
    return raw_ad
