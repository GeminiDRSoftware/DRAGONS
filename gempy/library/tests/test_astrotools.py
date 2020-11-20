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

    # scalar / scalar
    assert at.divide0(1, 0) == 0
    # array / scalar
    np.testing.assert_array_equal(at.divide0(ones, 0), zeros)
    # scalar / array
    np.testing.assert_array_equal(at.divide0(1, zeros), zeros)
    # array / array
    np.testing.assert_array_equal(at.divide0(ones, zeros), zeros)


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


def test_cartesian_regions_to_slices():
    cart = at.cartesian_regions_to_slices
    assert cart('')[0] == slice(None)
    assert cart('1:10')[0] == slice(0, 10)
    assert cart('[1:10]')[0] == slice(0, 10)
    assert cart('1:10:2')[0] == slice(0, 10, 2)
    assert cart('1-10:2')[0] == slice(0, 10, 2)
    assert cart('1:10,20:30') == (slice(19, 30), slice(0, 10))
    assert cart('[:,:10]') == (slice(None, 10), slice(None))

    with pytest.raises(TypeError):
        cart(12)
