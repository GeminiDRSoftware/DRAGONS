import pytest

import numpy as np

from gempy.library.telluric_models import ArrayInterpolator


def test_array_interpolator_1d_1par():
    """Confirm this works for a 1D array with one parameter"""
    values = np.arange(30).reshape(3, 10)
    ai = ArrayInterpolator(([0, 5, 10],), values)
    assert np.allclose(ai([0]), values[0])
    assert np.allclose(ai([4]), values[0] + 8)


def test_array_interpolator_1d_2par():
    """Confirm this works for a 1D array with two parameters"""
    values = np.arange(120).reshape(3, 4, 10)
    ai = ArrayInterpolator(([0, 5, 10], [-2, 0, 2, 4]), values)
    assert np.allclose(ai([0, 0]), values[0, 1])
    # 2/5 * 40 + 1/2 * 10 = 21
    assert np.allclose(ai([2, 1]), values[0, 1] + 21)


def test_array_interpolator_2d_1par():
    """Confirm this works for a 2D array with one parameter"""
    values = np.arange(120).reshape(3, 4, 10)
    ai = ArrayInterpolator(([0, 5, 10],), values)
    assert np.allclose(ai([0]), values[0])


def test_array_interpolator_2d_2par():
    """Confirm this works for a 2D array with two parameters"""
    values = np.arange(120).reshape(2, 3, 4, 5)
    ai = ArrayInterpolator(([0, 5], [-2, 0, 2]), values)
    assert np.allclose(ai([0, 0]), values[0, 1])
    # 2/5 * 60 + 1/2 * 20 = 34
    assert np.allclose(ai([2, 1]), values[0, 1] + 34)
