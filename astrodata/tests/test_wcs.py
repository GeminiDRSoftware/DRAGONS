import math
import pytest
from numpy.testing import assert_allclose

from astropy.modeling import models

from astrodata import wcs as adwcs


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
