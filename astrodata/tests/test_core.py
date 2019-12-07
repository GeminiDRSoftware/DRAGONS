#!/usr/bin/env python

import numpy as np
import operator
import pytest
from astropy.io import fits
from numpy.testing import assert_array_equal

import astrodata

SHAPE = (4, 5)


@pytest.fixture
def ad1():
    hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
    phu = fits.PrimaryHDU(header=hdr)
    hdu = fits.ImageHDU(data=np.ones(SHAPE), name='SCI')
    return astrodata.create(phu, [hdu])


@pytest.fixture
def ad2():
    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=np.ones(SHAPE) * 2, name='SCI')
    return astrodata.create(phu, [hdu])


def test_can_create_new_astrodata_object_with_no_data():
    ad = astrodata.create(fits.PrimaryHDU())
    assert isinstance(ad, astrodata.AstroData)
    assert len(ad) == 0
    assert ad.instrument() is None
    assert ad.object() is None


def test_can_create_astrodata_from_image_hdu():
    data_array = np.zeros(SHAPE)

    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=data_array, name='SCI')
    ad = astrodata.create(phu, [hdu])

    assert isinstance(ad, astrodata.AstroData)
    assert len(ad) == 1
    assert isinstance(ad[0].data, np.ndarray)
    assert ad[0].data is hdu.data


def test_can_append_an_image_hdu_object_to_an_astrodata_object():
    ad = astrodata.create(fits.PrimaryHDU())
    hdu = fits.ImageHDU(data=np.zeros(SHAPE))
    ad.append(hdu, name='SCI')
    ad.append(hdu, name='SCI2')

    assert len(ad) == 2
    assert ad[0].data is hdu.data
    assert ad[1].data is hdu.data


def test_attributes(ad1):
    assert ad1[0].shape == SHAPE

    data = ad1[0].data
    assert_array_equal(data, 1)
    assert data.mean() == 1
    assert np.median(data) == 1

    assert ad1.phu['INSTRUME'] == 'darkimager'
    assert ad1.instrument() == 'darkimager'
    assert ad1.object() == 'M42'


@pytest.mark.parametrize('op, res', [(operator.add, 3),
                                     (operator.sub, -1),
                                     (operator.mul, 2),
                                     (operator.divide, 0.5)])
def test_can_add_two_astrodata_objects(op, res, ad1, ad2):
    result = op(ad1, ad2)
    assert_array_equal(result.data, res)
    assert isinstance(result, astrodata.AstroData)
    assert len(result) == 1
    assert isinstance(result[0].data, np.ndarray)
    assert isinstance(result[0].hdr, fits.Header)


@pytest.mark.parametrize('op, arg, res', [('add', 100, 101),
                                          ('subtract', 100, -99),
                                          ('multiply', 3, 3),
                                          ('divide', 2, 0.5)])
def test_operations(ad1, op, arg, res):
    result = getattr(ad1, op)(arg)
    assert_array_equal(result.data, res)
    assert isinstance(result, astrodata.AstroData)
    assert isinstance(result[0].data, np.ndarray)
    assert isinstance(result[0].hdr, fits.Header)
    assert len(result) == 1


if __name__ == '__main__':
    pytest.main()
