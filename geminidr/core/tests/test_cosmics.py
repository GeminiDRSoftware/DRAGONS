import numpy as np
import pytest
from numpy.testing import assert_array_equal

from geminidr.gemini.lookups import DQ_definitions as DQ
from geminidr.niri.primitives_niri_image import NIRIImage


def add_fake_image(ext):
    # Add sky and sky noise
    ext.data += 200

    # Add some fake sources
    for i in range(100):
        x = np.random.uniform(low=0.0, high=1001)
        y = np.random.uniform(low=0.0, high=1001)
        amplitude = np.random.uniform(low=10., high=300.)
        if i % 10 == 0:
            ext.add_galaxy(amplitude=amplitude, x=x, y=y)
        else:
            ext.add_star(amplitude=amplitude, x=x, y=y)

    ext.add_poisson_noise()
    ext.add_read_noise()

    # Add 100 fake cosmic rays
    cr_x = np.random.randint(low=5, high=995, size=100)
    cr_y = np.random.randint(low=5, high=995, size=100)
    cr_brightnesses = np.random.uniform(low=1000.0, high=30000.0, size=100)
    ext.data[cr_y, cr_x] += cr_brightnesses

    ext /= ext.gain()

    # Make a mask where the detected cosmic rays should be
    ext.CRMASK = np.zeros(ext.shape, dtype=np.uint8)
    ext.CRMASK[cr_y, cr_x] = 1


@pytest.fixture(scope='module')
def adinputs():
    astrofaker = pytest.importorskip('astrofaker')
    np.random.seed(200)

    ad = astrofaker.create('NIRI', 'IMAGE')
    ad.init_default_extensions()
    add_fake_image(ad[0])

    return [ad]


def test_flag_cosmics(adinputs):
    p = NIRIImage(adinputs)
    adout = p.flagCosmicRays()[0]
    assert_array_equal(adout[0].mask == DQ.cosmic_ray,
                       adout[0].CRMASK.astype(bool))
