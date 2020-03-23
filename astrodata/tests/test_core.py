import operator
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import astrodata
from astropy.io import fits
from astropy.nddata import NDData, VarianceUncertainty

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


def test_create_with_no_data():
    for phu in (fits.PrimaryHDU(), fits.Header(), {}):
        ad = astrodata.create(phu)
        assert isinstance(ad, astrodata.AstroData)
        assert len(ad) == 0
        assert ad.instrument() is None
        assert ad.object() is None


def test_create_with_header():
    hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
    for phu in (hdr, fits.PrimaryHDU(header=hdr), dict(hdr), list(hdr.cards)):
        ad = astrodata.create(phu)
        assert isinstance(ad, astrodata.AstroData)
        assert len(ad) == 0
        assert ad.instrument() == 'darkimager'
        assert ad.object() == 'M42'


def test_create_from_hdu():
    phu = fits.PrimaryHDU()
    hdu = fits.ImageHDU(data=np.zeros(SHAPE), name='SCI')
    ad = astrodata.create(phu, [hdu])

    assert isinstance(ad, astrodata.AstroData)
    assert len(ad) == 1
    assert isinstance(ad[0].data, np.ndarray)
    assert ad[0].data is hdu.data


def test_create_invalid():
    with pytest.raises(ValueError):
        astrodata.create('FOOBAR')
    with pytest.raises(ValueError):
        astrodata.create(42)


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


@pytest.mark.parametrize('op, res, res2', [
    (operator.add, 3, 3),
    (operator.sub, -1, 1),
    (operator.mul, 2, 2),
    (operator.truediv, 0.5, 2)
])
def test_arithmetic(op, res, res2, ad1, ad2):
    for data in (ad2, ad2.data):
        result = op(ad1, data)
        assert_array_equal(result.data, res)
        assert isinstance(result, astrodata.AstroData)
        assert len(result) == 1
        assert isinstance(result[0].data, np.ndarray)
        assert isinstance(result[0].hdr, fits.Header)

        result = op(data, ad1)
        assert_array_equal(result.data, res2)

    for data in (ad2[0], ad2[0].data):
        result = op(ad1[0], data)
        assert_array_equal(result.data, res)
        assert isinstance(result, astrodata.AstroData)
        assert len(result) == 1
        assert isinstance(result[0].data, np.ndarray)
        assert isinstance(result[0].hdr, fits.Header)

    # FIXME: should work also with ad2[0].data, but crashes
    result = op(ad2[0], ad1[0])
    assert_array_equal(result.data, res2)


@pytest.mark.parametrize('op, res, res2', [
    (operator.iadd, 3, 3),
    (operator.isub, -1, 1),
    (operator.imul, 2, 2),
    (operator.itruediv, 0.5, 2)
])
def test_arithmetic_inplace(op, res, res2, ad1, ad2):
    for data in (ad2, ad2.data):
        ad = deepcopy(ad1)
        op(ad, data)
        assert_array_equal(ad.data, res)
        assert isinstance(ad, astrodata.AstroData)
        assert len(ad) == 1
        assert isinstance(ad[0].data, np.ndarray)
        assert isinstance(ad[0].hdr, fits.Header)

    # data2 = deepcopy(ad2[0])
    # op(data2, ad1)
    # assert_array_equal(data2, res2)

    for data in (ad2[0], ad2[0].data):
        ad = deepcopy(ad1)
        op(ad[0], data)
        assert_array_equal(ad.data, res)
        assert isinstance(ad, astrodata.AstroData)
        assert len(ad) == 1
        assert isinstance(ad[0].data, np.ndarray)
        assert isinstance(ad[0].hdr, fits.Header)


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


def test_operate():
    ad = astrodata.create({})
    nd = NDData(data=[[1, 2], [3, 4]],
                uncertainty=VarianceUncertainty(np.ones((2, 2))),
                mask=np.identity(2), meta={'header': {}})
    ad.append(nd)

    ad.operate(np.sum, axis=1)
    assert_array_equal(ad[0].data, [3, 7])
    assert_array_equal(ad[0].variance, [2, 2])
    assert_array_equal(ad[0].mask, [1, 1])


def test_reset(ad1):
    data = np.ones(SHAPE) + 3
    with pytest.raises(ValueError):
        ad1.reset(data)

    ext = ad1[0]

    with pytest.raises(ValueError):
        ext.reset(data, mask=np.ones((2, 2)), check=True)

    with pytest.raises(ValueError):
        ext.reset(data, variance=np.ones((2, 2)), check=True)

    ext.reset(data, mask=np.ones(SHAPE), variance=np.ones(SHAPE))
    assert_array_equal(ext.data, 4)
    assert_array_equal(ext.variance, 1)
    assert_array_equal(ext.mask, 1)

    ext.reset(data, mask=None, variance=None)
    assert ext.mask is None
    assert ext.uncertainty is None

    ext.reset(np.ma.array(data, mask=np.ones(SHAPE)))
    assert_array_equal(ext.data, 4)
    assert_array_equal(ext.mask, 1)

    with pytest.raises(TypeError):
        ext.reset(data, mask=1)

    with pytest.raises(TypeError):
        ext.reset(data, variance=1)

    nd = NDData(data=data,
                uncertainty=VarianceUncertainty(np.ones((2, 2)) * 3),
                mask=np.ones((2, 2)) * 2, meta={'header': {}})
    ext.reset(nd)
    assert_array_equal(ext.data, 4)
    assert_array_equal(ext.variance, 3)
    assert_array_equal(ext.mask, 2)
