import pytest

from astropy.modeling import models
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io.fits import Header

import numpy as np
from geminidr.gmos.primitives_gmos_image import GMOSImage
from gempy.library import transform
from geminidr.gmos.lookups import geometry_conf as geotable

# TODO: Either remove astrodata dependency or move the
# transform module to gempy/adlibrary.  That includes all
# the functions that accept an AD object as input, not just
# the need to import astrodata.
import astrodata
from astrodata.testing import download_from_archive

@pytest.fixture(scope='module')
def GMOS_LONGSLIT():
    """Any GMOS longslit spectrum"""
    return download_from_archive("N20180103S0332.fits")

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


@pytest.mark.dragons_remote_data
def test_adding_longslit_wcs(GMOS_LONGSLIT):
    """Test that adding the longslit WCS doesn't interfere with the sky
    coordinates of the WCS"""
    ad = astrodata.open(GMOS_LONGSLIT)
    frame_name = ad[4].hdr.get("RADESYS", ad[4].hdr["RADECSYS"]).lower()
    crpix1 = ad[4].hdr["CRPIX1"] - 1
    crpix2 = ad[4].hdr["CRPIX2"] - 1
    gwcs_sky = SkyCoord(*ad[4].wcs(crpix1, crpix2), unit=u.deg, frame=frame_name)
    transform.add_longslit_wcs(ad)
    gwcs_coords = ad[4].wcs(crpix1, crpix2)
    new_gwcs_sky = SkyCoord(*gwcs_coords[1:], unit=u.deg, frame=frame_name)
    assert gwcs_sky.separation(new_gwcs_sky) < 0.01 * u.arcsec
    # The sky coordinates should not depend on the x pixel value
    gwcs_coords = ad[4].wcs(0, crpix2)
    new_gwcs_sky = SkyCoord(*gwcs_coords[1:], unit=u.deg, frame=frame_name)
    assert gwcs_sky.separation(new_gwcs_sky) < 0.01 * u.arcsec

    # The sky coordinates also should not depend on the extension
    # there are shifts of order 1 pixel because of the rotations of CCDs 1
    # and 3, which are incorporated into their raw WCSs. Remember that the
    # 12 WCSs are independent at this stage, they don't all map onto the
    # WCS of the reference extension
    for ext in ad:
        gwcs_coords = ext.wcs(0, crpix2)
        new_gwcs_sky = SkyCoord(*gwcs_coords[1:], unit=u.deg, frame=frame_name)
        assert gwcs_sky.separation(new_gwcs_sky) < 0.1 * u.arcsec

    # This is equivalent to writing to disk and reading back in
    wcs_dict = astrodata.wcs.gwcs_to_fits(ad[4].nddata, ad.phu)
    new_gwcs = astrodata.wcs.fitswcs_to_gwcs(Header(wcs_dict))
    gwcs_coords = new_gwcs(crpix1, crpix2)
    new_gwcs_sky = SkyCoord(*gwcs_coords[1:], unit=u.deg, frame=frame_name)
    assert gwcs_sky.separation(new_gwcs_sky) < 0.01 * u.arcsec
    gwcs_coords = new_gwcs(0, crpix2)
    new_gwcs_sky = SkyCoord(*gwcs_coords[1:], unit=u.deg, frame=frame_name)
    assert gwcs_sky.separation(new_gwcs_sky) < 0.01 * u.arcsec
