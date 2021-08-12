import math
import pytest
from numpy.testing import assert_allclose

from astropy.modeling import models

import astrodata
from astrodata import wcs as adwcs
from astrodata.testing import download_from_archive


@pytest.fixture(scope='module')
def F2_IMAGE():
    """Any F2 image with CD3_3=1"""
    return download_from_archive("S20130717S0365.fits")


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
def test_reading_and_writing_sliced_image(F2_IMAGE):
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


def test_remove_axis_from_model_1():
    """A simple test that removes one of three &-linked models"""
    model = models.Shift(0) & models.Shift(1) & models.Shift(2)
    for axis in (0, 1, 2):
        new_model, input_axis = adwcs.remove_axis_from_model(model, axis)
        assert input_axis == axis
        assert new_model.n_submodels == 2
        assert new_model.offset_0 + new_model.offset_1 == 3 - axis


def test_remove_axis_from_model_2():
    """A test with |-chained models"""
    model = ((models.Shift(0) & models.Shift(1) & models.Shift(2)) |
             (models.Scale(2) & models.Rotation2D(90)))
    new_model, input_axis = adwcs.remove_axis_from_model(model, 0)
    assert input_axis == 0
    assert new_model.n_submodels == 3
    assert new_model.offset_0 == 1
    assert new_model.offset_1 == 2
    assert new_model.angle_2 == 90


def test_remove_axis_from_model_3():
    """A test with a Mapping"""
    model1 = models.Mapping((1, 2, 0))
    model2 = models.Shift(0) & models.Shift(1) & models.Shift(2)
    new_model, input_axis = adwcs.remove_axis_from_model(model1 | model2, 1)
    assert input_axis == 2
    assert new_model.n_submodels == 3
    assert_allclose(new_model(0, 10), (10, 2))
    new_model, input_axis = adwcs.remove_axis_from_model(model2 | model1, 1)
    assert input_axis == 2
    assert new_model.n_submodels == 3
    assert_allclose(new_model(0, 10), (11, 0))


def test_remove_axis_from_model_4():
    """A test with a Mapping that creates a new axis"""
    model1 = models.Shift(0) & models.Shift(1) & models.Shift(2)
    model = models.Mapping((1, 0, 0)) | model1
    new_model, input_axis = adwcs.remove_axis_from_model(model, 1)
    assert input_axis is None
    assert new_model.n_submodels == 2
    assert_allclose(new_model(0, 10), (10, 2))

    # Check that we can identify and remove the "Identity"-like residual Mapping
    model = models.Mapping((0, 1, 0)) | model1
    new_model, input_axis = adwcs.remove_axis_from_model(model, 2)
    assert input_axis is None
    assert new_model.n_submodels == 2
    assert_allclose(new_model(0, 10), (0, 11))


def test_remove_axis_from_model_5():
    """A test with fix_inputs"""
    model1 = models.Shift(0) & models.Shift(1) & models.Shift(2)
    model = models.fix_inputs(model1, {1: 6})
    new_model, input_axis = adwcs.remove_axis_from_model(model, 1)
    assert input_axis is None
    assert new_model.n_submodels == 2
    assert_allclose(new_model(0, 10), (0, 12))

    new_model, input_axis = adwcs.remove_axis_from_model(model, 2)
    assert input_axis == 2
    assert new_model.n_submodels == 3
    assert_allclose(new_model(0), (0, 7))


