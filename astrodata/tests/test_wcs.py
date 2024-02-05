import math
import pytest
import numpy as np
from numpy.testing import assert_allclose

from astropy.modeling import models
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.io.fits import Header
from gwcs import coordinate_frames as cf
from gwcs.wcs import WCS as gWCS

import astrodata
from astrodata import wcs as adwcs
from astrodata.testing import download_from_archive
from gempy.library.transform import add_longslit_wcs


@pytest.fixture(scope='module')
def F2_IMAGE():
    """Any F2 image with CD3_3=1"""
    return download_from_archive("S20130717S0365.fits")


@pytest.fixture(scope='module')
def NIRI_IMAGE():
    """Any NIRI image"""
    return download_from_archive("N20180102S0392.fits")


@pytest.fixture(scope='module')
def GMOS_LONGSLIT():
    """Any GMOS longslit spectrum"""
    return download_from_archive("N20180103S0332.fits")


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


def test_remove_axis_from_model():
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
    assert new_model.n_submodels == 3
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


@pytest.mark.dragons_remote_data
def test_remove_unused_world_axis(F2_IMAGE):
    """A test with an intermediate frame"""
    ad = astrodata.open(F2_IMAGE)
    result = ad[0].wcs(1000, 1000, 0)
    new_frame = cf.Frame2D(name="intermediate")
    new_model = models.Shift(100) & models.Shift(200) & models.Identity(1)
    ad[0].wcs.insert_frame(ad[0].wcs.input_frame,
                           new_model, new_frame)
    ad[0].reset(ad[0].nddata[0])
    new_result = ad[0].wcs(900, 800)
    assert_allclose(new_result, result)
    adwcs.remove_unused_world_axis(ad[0])
    new_result = ad[0].wcs(900, 800)
    assert_allclose(new_result, result[:2])
    for frame in ad[0].wcs.available_frames:
        assert getattr(ad[0].wcs, frame).naxes == 2


@pytest.mark.dragons_remote_data
def test_gwcs_creation(NIRI_IMAGE):
    """Test that the gWCS object for an image agrees with the FITS WCS"""
    ad = astrodata.open(NIRI_IMAGE)
    w = WCS(ad[0].hdr)
    for y in range(0, 1024, 200):
        for x in range(0, 1024, 200):
            wcs_sky = w.pixel_to_world(x, y)
            gwcs_sky = SkyCoord(*ad[0].wcs(x, y), unit=u.deg)
            assert wcs_sky.separation(gwcs_sky) < 0.01 * u.arcsec


@pytest.mark.dragons_remote_data
def test_adding_longslit_wcs(GMOS_LONGSLIT):
    """Test that adding the longslit WCS doesn't interfere with the sky
    coordinates of the WCS"""
    ad = astrodata.open(GMOS_LONGSLIT)
    frame_name = ad[4].hdr.get("RADESYS", ad[4].hdr["RADECSYS"]).lower()
    crpix1 = ad[4].hdr["CRPIX1"] - 1
    crpix2 = ad[4].hdr["CRPIX2"] - 1
    gwcs_sky = SkyCoord(*ad[4].wcs(crpix1, crpix2), unit=u.deg, frame=frame_name)
    add_longslit_wcs(ad)
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


@pytest.mark.dragons_remote_data
def test_loglinear_axis(NIRI_IMAGE):
    """Test that we can add a log-linear axis and write and read it"""
    ad = astrodata.open(NIRI_IMAGE)
    coords = ad[0].wcs(200, 300)
    ad[0].data = np.repeat(ad[0].data[:, :, np.newaxis], 5, axis=2)
    new_input_frame = adwcs.pixel_frame(3)
    loglinear_frame = cf.SpectralFrame(axes_order=(0,), unit=u.nm,
                                 axes_names=("AWAV",), name="Wavelength in air")
    celestial_frame = ad[0].wcs.output_frame
    celestial_frame._axes_order = (1, 2)
    new_output_frame = cf.CompositeFrame([loglinear_frame, celestial_frame],
                                         name="world")
    new_wcs = models.Exponential1D(amplitude=1, tau=2) & ad[0].wcs.forward_transform
    ad[0].wcs = gWCS([(new_input_frame, new_wcs),
                      (new_output_frame, None)])
    new_coords = ad[0].wcs(2, 200, 300)
    assert_allclose(coords, new_coords[1:])

    #with change_working_dir():
    ad.write("test.fits", overwrite=True)
    ad2 = astrodata.open("test.fits")
    assert_allclose(ad2[0].wcs(2, 200, 300), new_coords)
