"""
Tests applied to primitives_ccd.py
"""

from astropy.io import fits
import numpy as np
import pytest

import astrodata, gemini_instruments
from astrodata.testing import download_from_archive
from geminidr.gmos.primitives_gmos import GMOS
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


@pytest.mark.parametrize("rejection_method, expected",
                         [('varclip', 2.),
                          ('sigclip', 1.6),
                          ('minmax', 1.666666)])
def test_stack_biases(rejection_method, expected, fake_biases):

    p = GMOS(fake_biases)
    p.addVAR()
    if rejection_method == 'minmax':
        ad_out = p.stackBiases(reject_method=rejection_method, nlow=1, nhigh=1)
    else:
        ad_out = p.stackBiases(reject_method=rejection_method)
    assert(len(ad_out)) == 1
    assert pytest.approx(ad_out[0].data[0]) == expected


def test_non_bias_mixed_in(fake_biases):

    # Remove BIAS tag from one fake input astrodata object
    fake_biases[0].tags = fake_biases[0].tags.difference({'BIAS'})
    p = GMOS(fake_biases)

    with pytest.raises(OSError, match='Not all inputs have BIAS tag'):
        p.stackBiases()


def create_bias(height, width, value):

    astrofaker = pytest.importorskip("astrofaker")

    data = np.ones((height, width)) * value
    hdu = fits.ImageHDU()
    hdu.data = data

    ad = astrofaker.create('GMOS-N')
    ad.add_extension(hdu, pixel_scale=1)
    ad.tags = ad.tags.union({'BIAS'})
    for ext in ad:
        ext.variance = np.where(ext.data > 0,
                                ext.data, 0).astype(np.float32)

    return ad


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture(scope='function')
def raw_ad(request):
    filename = request.param
    raw_ad = astrodata.open(download_from_archive(filename))
    return raw_ad

# Create a set of fake biases
@pytest.fixture(scope='module')
def fake_biases():
    return [create_bias(100, 100, i) for i in (0, 1, 2, 2, 3)]


if __name__ == '__main__':
    pytest.main()

