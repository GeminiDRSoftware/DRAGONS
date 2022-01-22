"""
Tests applied to primitives_ccd.py
"""

from astropy.io import fits
import numpy as np
import pytest

from geminidr.gmos.primitives_gmos import GMOS

# -- Tests --------------------------------------------------------------------


@pytest.fixture()
def fake_biases():
    return [create_bias(100, 100) for i in range(5)]

@pytest.mark.parametrize("rejection_method", ('varclip', 'sigclip', 'minmax'))
def test_stack_biases(rejection_method, fake_biases):

    p = GMOS(fake_biases)
    p.addVAR()
    ad_out = p.stackBiases(reject_method=rejection_method)
    assert(len(ad_out)) == 1
    assert ad_out[0].data[0][0, 0] == 1.


def test_non_bias_mixed_in(fake_biases):

    # Remove BIAS tag from one fake input astrodata object
    fake_biases[0].tags = fake_biases[0].tags.difference({'BIAS'})
    p = GMOS(fake_biases)

    with pytest.raises(OSError, match='Not all inputs have BIAS tag'):
        p.stackBiases()


def create_bias(height, width):

    astrofaker = pytest.importorskip("astrofaker")

    data = np.ones((height, width))
    hdu = fits.ImageHDU()
    hdu.data = data

    ad = astrofaker.create('GMOS-N')
    ad.add_extension(hdu, pixel_scale=1)
    ad.tags = ad.tags.union({'BIAS'})

    return ad


if __name__ == '__main__':
    pytest.main()
