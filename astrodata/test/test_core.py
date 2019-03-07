
import copy
import numpy as np
import pytest

from astropy.io import fits

import astrodata


@pytest.fixture
def sample_astrodata_with_ones():

    data_array = np.ones((100, 100))

    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=data_array, name='SCI')

    ad = astrodata.create(phu, [hdu])

    return ad


def test_can_create_new_astrodata_object_with_no_data():

    primary_header_unit = fits.PrimaryHDU()

    ad = astrodata.create(primary_header_unit)

    assert isinstance(ad, astrodata.AstroData)


def test_can_create_astrodata_from_image_hdu():

    data_array = np.zeros((100, 100))

    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=data_array, name='SCI')

    ad = astrodata.create(phu, [hdu])

    assert isinstance(ad, astrodata.AstroData)
    assert len(ad) == 1
    assert isinstance(ad[0].data, np.ndarray)


def test_can_append_an_image_hdu_object_to_an_astrodata_object():

    primary_header_unit = fits.PrimaryHDU()

    ad = astrodata.create(primary_header_unit)

    data_array = np.zeros((100, 100))

    header_data_unit = fits.ImageHDU()
    header_data_unit.data = data_array

    ad.append(header_data_unit, name='SCI')
    ad.append(header_data_unit, name='SCI2')

    assert len(ad) == 2


def test_can_add_two_astrodata_objects(sample_astrodata_with_ones):

    ad1 = copy.deepcopy(sample_astrodata_with_ones)
    ad2 = copy.deepcopy(sample_astrodata_with_ones)

    result = ad1 + ad2

    expected = np.ones((100, 100)) * 2

    assert isinstance(result, astrodata.AstroData)
    assert len(result) == 1
    assert isinstance(result[0].data, np.ndarray)
    assert isinstance(result[0].hdr, fits.Header)

    np.testing.assert_array_almost_equal(result[0].data, expected)


def test_can_subtract_two_astrodata_objects(sample_astrodata_with_ones):

    ad1 = copy.deepcopy(sample_astrodata_with_ones)
    ad2 = copy.deepcopy(sample_astrodata_with_ones)

    result = ad1 - ad2

    expected = np.zeros((100, 100))

    assert isinstance(result, astrodata.AstroData)
    assert len(result) == 1
    assert isinstance(result[0].data, np.ndarray)
    assert isinstance(result[0].hdr, fits.Header)

    np.testing.assert_array_almost_equal(result[0].data, expected)
