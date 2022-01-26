"""
Tests for the astrotools module.
"""

import numpy as np
import pytest
from gempy.library import astrotools as at
from astropy import units as u


def test_array_from_list():
    values = (1, 2, 3)
    unit = u.m
    x = [i * unit for i in values]
    result = at.array_from_list(x)
    assert result.unit is unit
    np.testing.assert_array_equal(result.value, values)


def test_divide0():
    ones = np.array([1, 1, 1])
    zeros = np.array([0, 0, 0])
    twod = np.arange(12).reshape(4, 3)

    # scalar / scalar
    assert at.divide0(1, 0) == 0
    # array / scalar
    np.testing.assert_array_equal(at.divide0(ones, 0), zeros)
    # scalar / array
    np.testing.assert_array_equal(at.divide0(1, zeros), zeros)
    # array / array
    np.testing.assert_array_equal(at.divide0(ones, zeros), zeros)

    # 2d array / 1d array
    np.testing.assert_array_equal(at.divide0(twod, ones), twod)
    np.testing.assert_array_equal(at.divide0(twod, zeros), np.zeros_like(twod))


def test_fit_spline_to_data():
    rng = np.random.default_rng(0)
    x = np.arange(100)
    y = np.ones_like(x)
    spline = at.fit_spline_to_data(y)
    np.testing.assert_allclose(spline(x), y)

    # Test with a curve and noise
    y = x + x*x + rng.normal(size=x.size)
    spline = at.fit_spline_to_data(y)
    np.testing.assert_allclose(spline(x), y, atol=3)

    # Test with some masking
    mask = np.zeros_like(x, dtype=bool)
    safe = y[10]
    y[10] = 1000
    mask[10] = True
    spline = at.fit_spline_to_data(y, mask=mask)
    y[10] = safe
    np.testing.assert_allclose(spline(x), y, atol=3)


@pytest.mark.parametrize("separation", [1,2,3,4,5])
def test_std_from_pixel_variations(separation):
    # Test passes with ths seed and number of samples
    rng = np.random.default_rng(1)
    data = rng.normal(size=10000)
    assert abs(at.std_from_pixel_variations(data, separation=separation) - 1) < 0.02


def test_rasextodec():
    rastring = '20:30:40.506'
    ra = at.rasextodec(rastring)
    assert abs(ra - 307.668775) < 0.0001


def test_degsextodec():
    decstringneg = '-60:50:40.302'
    decstringpos = '60:50:40.302'
    decneg = at.degsextodec(decstringneg)
    decpos = at.degsextodec(decstringpos)
    assert abs(decneg + decpos - 0.) < 0.0001


def test_get_corners_2d():
    corners = at.get_corners((300, 500))
    assert corners == [(0, 0), (299, 0), (0, 499), (299, 499)]


def test_get_corners_3d():
    corners = at.get_corners((300, 500, 400))
    expected_corners = [(0, 0, 0), (299, 0, 0), (0, 499, 0),
                        (299, 499, 0), (0, 0, 399), (299, 0, 399),
                        (0, 499, 399), (299, 499, 399)]
    assert corners == expected_corners


def test_rotate_2d():
    rotation_matrix = at.rotate_2d(30.)
    expected_matrix = np.array([[0.8660254, -0.5],
                                [0.5, 0.8660254]])
    assert np.allclose(rotation_matrix, expected_matrix)


def test_clipped_mean():
    dist = np.array([5, 5, 4, 7, 7, 4, 3, 5, 2, 6, 5, 12, 0,
                     9, 10, 13, 2, 14, 6, 3, 50])
    results = at.clipped_mean(dist)
    expected_values = (6.1, 3.7)
    assert np.allclose(results, expected_values)


def test_parse_user_regions():
    parse = at.parse_user_regions
    assert parse("*") == [(None, None)]
    assert parse("") == [(None, None)]
    assert parse(None) == [(None, None)]
    assert parse("1:10,20:50") == [(1, 10), (20, 50)]
    assert parse("1:10.2,20:50.2", dtype=float) == [(1, 10.2), (20, 50.2)]
    assert parse("1:10,20:50:2", allow_step=True) == [(1, 10), (20, 50, 2)]
    with pytest.raises(ValueError):
        parse("1:10:2")
    assert parse("50:20") == [(20, 50), ]


def test_cartesian_regions_to_slices():
    cart = at.cartesian_regions_to_slices
    assert cart('')[0] == slice(None)
    assert cart('1:10')[0] == slice(0, 10)
    assert cart('[1:10]')[0] == slice(0, 10)
    assert cart('1:10:2')[0] == slice(0, 10, 2)
    assert cart('1-10:2')[0] == slice(0, 10, 2)
    assert cart('1:10,20:30') == (slice(19, 30), slice(0, 10))
    assert cart('[:,:10]') == (slice(None, 10), slice(None))

    assert cart('100,*') == (slice(None), slice(99, 100))
    assert cart('100, *') == (slice(None), slice(99, 100))
    assert cart(':100,*') == (slice(None), slice(100))

    assert cart('*,12') == (slice(11, 12), slice(None))
    assert cart('*, 12:') == (slice(11, None), slice(None))

    with pytest.raises(TypeError):
        cart(12)
