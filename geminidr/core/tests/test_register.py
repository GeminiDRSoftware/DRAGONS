"""
Tests for primitives_register
"""

import numpy as np
import pytest
from astropy.modeling import models
from geminidr.niri.primitives_niri_image import NIRIImage

# xoffset, yoffset, angle, scale for models
MODEL_PARMS = ((-8, 12, 1.0, 0.99), (3, -10, -0.15, 1.02))


def make_images(astrofaker, mods, nstars=20):
    """
    Create a series of AstroFakerNiri images with a set number of stars
    in random locations, but offset in each image according to a series
    of Model instances.
    """
    np.random.seed(0)
    star_positions = np.random.rand(2, nstars) * 924 + 100
    adinputs = []
    for i, m in enumerate(mods, start=1):
        ad = astrofaker.create('NIRI', filename=f'test{i}.fits')
        ad.init_default_extensions()

        x, y = m.inverse(*star_positions)
        ad[0].add_stars(amplitude=1000, x=x, y=y, n_models=nstars)

        # Add some extra stars that aren't correlated
        extra_star_positions = np.random.rand(2, int(0.25 * nstars)) * 1024
        x, y = extra_star_positions
        ad[0].add_stars(amplitude=1000, x=x, y=y, n_models=len(x))

        ad.add_read_noise()
        adinputs.append(ad)
    return adinputs


@pytest.mark.parametrize('no_wcs', (False, True))
@pytest.mark.parametrize('rotate', (False, True))
@pytest.mark.parametrize('scale', (False, True))
def test_adjust_wcs_to_reference(astrofaker, no_wcs, rotate, scale):
    scale = False
    mods = [models.Identity(2)]
    for params in MODEL_PARMS:
        m = models.Shift(params[0]) & models.Shift(params[1])
        if rotate:
            m |= models.Rotation2D(params[2])
        if scale:
            m |= models.Scale(params[3]) & models.Scale(params[3])
        mods.append(m)
    adinputs = make_images(astrofaker, mods=mods)
    if no_wcs:
        for ad in adinputs:
            del ad[0].hdr['CD*']
            ad[0].wcs = None
    p = NIRIImage(adinputs)
    p.detectSources()
    p.adjustWCSToReference(first_pass=42 if no_wcs else 5,
                           rotate=rotate, scale=scale)

    # We're going to confirm that a grid of input points are transformed
    # correctly, rather than comparing the components of the registration
    # model with the original model
    yin, xin = np.mgrid[:1000:100, :1000:100]
    for i, (ad, m) in enumerate(zip(p.streams['main'], mods)):
        if i == 0:  # the Identity model: nothing to check
            continue
        m_regist = ad[0].wcs.forward_transform[:m.n_submodels]
        print(m_regist)

        xref, yref = m(xin, yin)
        xout, yout = m_regist(xin, yin)
        np.testing.assert_allclose(xref, xout, atol=0.1)
        np.testing.assert_allclose(yref, yout, atol=0.1)
        rms = np.sqrt(((xref-xout)**2 + (yref-yout)**2).mean())
        assert rms < 0.05
