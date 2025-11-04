import pytest

from astropy.modeling import models

import numpy as np
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.library import transform
from geminidr.gmos.lookups import geometry_conf as geotable

@models.custom_model
def InverseQuadratic1D(x, c0=0, c1=0, c2=0):
    return 0.5 * (np.sqrt(c1*c1 - 4*c2*(c0-x)) - c1) / c2

# Star locations: Use unique y values to enable sorting
GMOS_STAR_LOCATIONS = ((200, 50), (204, 450), (4000, 50), (4004, 450))


# This functionality (which isn't used in DRAGONS) is no longer available
@pytest.mark.skip("Functionality lost in refactor")
@pytest.mark.parametrize('binning', (1, 2, 4))
def test_inverse_transform_gmos(astrofaker, binning):
    # Creates GMOS images with stars at predefined points
    ad = astrofaker.create('GMOS-N')
    ad.init_default_extensions(binning=binning, overscan=False)
    for ext in ad:
        ext.add(np.random.randn(*ext.shape))
        for ystar, xstar in GMOS_STAR_LOCATIONS:
            ext.add_star(amplitude=10000, x=xstar / binning, y=ystar / binning)

    adg = transform.create_mosaic_transform(ad, geotable)
    admos = adg.transform(attributes=None, order=1)
    adout = adg.inverse_transform(admos, order=3)
    p = GMOSImage([adout])
    p.detectSources()
    adout = p.streams['main'][0]
    xbin, ybin = ad.detector_x_bin(), ad.detector_y_bin()
    for ext in adout:
        objcat = ext.OBJCAT
        objcat.sort(['Y_IMAGE'])
        for row, location in zip(objcat, GMOS_STAR_LOCATIONS):
            # OBJCAT is 1-indexed
            assert abs(row['Y_IMAGE'] - location[0] / ybin - 1) < 0.1
            assert abs(row['X_IMAGE'] - location[1] / xbin - 1) < 0.1

def test_1d_affine_transform():
    """Test a simple 1D transform with and without flux conservation"""
    size = 100
    x = np.arange(size, dtype=np.float32)
    # The 0.5-pixel shifts ensure that each input pixel has a footprint
    # that precisely covers output pixels
    # We still lose the pixels at either end, resulting in a 198-pixel array
    m = models.Shift(0.5) | models.Scale(2) | models.Shift(-0.5)

    dg = transform.DataGroup([x], [transform.Transform(m)])

    output = dg.transform()
    y = output['data'][1:-1].reshape(size-2, 2).mean(axis=1)
    assert np.array_equal(x[1:-1], y)

    output = dg.transform(conserve=True)
    y = output['data'][1:-1].reshape(size-2, 2).sum(axis=1)
    assert np.array_equal(x[1:-1], y)


def test_2d_affine_transform():
    """Test a simple 2D transform with and without flux conservation"""
    size = 10
    x = np.arange(size * size, dtype=np.float32).reshape(size, size)
    # We still lose the pixels at either end, resulting in a 18x18-pixel array
    m = models.Shift(0.5) | models.Scale(2) | models.Shift(-0.5)

    dg = transform.DataGroup([x], [transform.Transform(m & m)])

    output = dg.transform()
    y = output['data'][1:-1, 1:-1].reshape(8,2,8,2).mean(axis=3).mean(axis=1)
    assert np.array_equal(x[1:-1, 1:-1], y)

    output = dg.transform(conserve=True)
    y = output['data'][1:-1, 1:-1].reshape(8,2,8,2).sum(axis=3).sum(axis=1)
    assert np.array_equal(x[1:-1, 1:-1], y)


def test_1d_nonaffine_transform():
    """Test a more complex 1D transform with and without flux conservation"""
    triangle = models.Polynomial1D(degree=2, c0=0, c1=0.5, c2=0.5)
    triangle.inverse = InverseQuadratic1D(c0=0, c1=0.5, c2=0.5)

    size = 100
    x = np.arange(size, dtype=np.float32)
    m = models.Shift(0.5) | triangle | models.Shift(-0.5)

    dg = transform.DataGroup([x], [transform.Transform(m)])

    output = dg.transform(conserve=True)
    y = [output['data'][i * (i + 1) // 2: (i + 1) * (i + 2) // 2].sum()
         for i in range(5, size-1)]
    assert np.allclose(x[5:-1], y, rtol=0.001)

    # The mean isn't really the right thing because the output pixels aren't
    # evenly spread within each input pixel, so larger differences are expected
    # but this test confirms that things aren't going completely nuts!
    output = dg.transform()
    # We start at 5 to avoid numerical issues of ~1% when limited resampling
    y = [output['data'][i * (i + 1) // 2: (i + 1) * (i + 2) // 2].mean()
         for i in range(5, size-1)]
    assert np.allclose(x[5:-1], y, rtol=0.005)


def test_2d_nonaffine_transform():
    """Test a more complex 2D transform with and without flux conservation"""
    triangle = models.Polynomial1D(degree=2, c0=0, c1=0.5, c2=0.5)
    triangle.inverse = InverseQuadratic1D(c0=0, c1=0.5, c2=0.5)

    size = 10
    x = np.arange(size * size, dtype=np.float32).reshape(size, size)
    m = models.Shift(0.5) | triangle | models.Shift(-0.5)

    dg = transform.DataGroup([x], [transform.Transform(m & m)])

    output = dg.transform(subsample=5, conserve=True)
    y = [output['data'][i * (i + 1) // 2: (i + 1) * (i + 2) // 2,
                        j * (j+1) // 2: (j + 1) * (j + 2) //2].sum()
         for i in range(1, size-1) for j in range(1, size-1)]
    assert np.allclose(x[1:-1, 1:-1], np.array(y).reshape(size-2, size-2), rtol=0.005)

    # As before, the mean won't give very good results. It's worse here
    # because the gradient is very high in one direction
    output = dg.transform()
    y = [output['data'][i * (i + 1) // 2: (i + 1) * (i + 2) // 2,
                        j * (j+1) // 2: (j + 1) * (j + 2) //2].mean()
         for i in range(1, size-1) for j in range(1, size-1)]
    assert np.allclose(x[1:-1, 1:-1], np.array(y).reshape(size-2, size-2),
                       rtol=0.03, atol=0.2)
