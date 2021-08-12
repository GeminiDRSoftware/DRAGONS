import math
import pytest
from numpy.testing import assert_allclose

from astropy.modeling import models

import astrodata
from astrodata import wcs as adwcs
from astrodata.testing import download_from_archive


@pytest.fixture(scope='module')
def F2_IMAGE():
    """
    No.    Name      Ver    Type      Cards   Dimensions   Format
      0  PRIMARY       1 PrimaryHDU     289   ()
      1                1 ImageHDU       144   (2048, 2048)   float32
      2                2 ImageHDU       144   (2048, 2048)   float32
      3                3 ImageHDU       144   (2048, 2048)   float32
      4                4 ImageHDU       144   (2048, 2048)   float32
    """
    return download_from_archive("S20150609S0023.fits")


@pytest.mark.parametrize("angle", [0, 20, 67, -35])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("xoffset,yoffset", [(0,0), (10,20)])
def test_calculate_affine_matrices(angle, scale, xoffset, yoffset):
    m = ((models.Scale(scale) & models.Scale(scale)) |
         models.Rotation2D(angle) |
         (models.Shift(xoffset) & models.Shift(yoffset)))
    affine = adwcs.calculate_affine_matrices(m, (100, 100))
    assert_allclose(affine.offset, (yoffset, xoffset), atol=1e-10)
    angle = math.radians(angle)
    assert_allclose(affine.matrix, ((scale * math.cos(angle), scale * math.sin(angle)),
                                    (-scale * math.sin(angle), scale * math.cos(angle))),
                    atol=1e-10)


@pytest.mark.dragons_remote_data
def test_reading_and_writing(F2_IMAGE):
    ad = astrodata.open(F2_IMAGE)
    result = ad[0].wcs(100, 100, 0)
    ad[0].reset(ad[0].nddata[0])
    assert_allclose(ad[0].wcs(100, 100), result)
    ad.write("test.fits", overwrite=True)
    ad2 = astrodata.open("test.fits")
    assert_allclose(ad2[0].wcs(100, 100), result)
    ad2.write("test.fits", overwrite=True)
    ad2 = astrodata.open("test.fits")
    assert_allclose(ad2[0].wcs(100, 100), result)
