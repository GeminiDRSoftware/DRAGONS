# pytest suite

"""
Tests for the astrotools module.

This is a suite of tests to be run with pytest.

To run:
   1) Set the environment variable GEMPYTHON_TESTDATA to the path that contains
      the file N20130510S0178_forStack.fits.
      Eg. /net/chara/data2/pub/ad_testdata/GMOS
   2) Then run: py.test -v   (must in gemini_python or have it in PYTHONPATH)
"""

# import os
# import os.path
import numpy as np
from gempy.library import astrotools as at
from astropy import units as u


# TESTDATAPATH = os.getenv('GEMPYTHON_TESTDATA', '.')
# TESTFITS = 'N20130510S0178_forStack.fits'

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
