"""
Tests applied to primitives_ccd.py
"""

from astropy.io import fits
import numpy as np
import pytest

from geminidr.gmos.primitives_gmos import GMOS

# -- Tests --------------------------------------------------------------------

# Create a set of fake biases
@pytest.fixture(scope='module')
def fake_biases():
    return [create_bias(100, 100, i) for i in (0, 1, 2, 2, 3)]

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


if __name__ == '__main__':
    pytest.main()
