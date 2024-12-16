import pytest

from functools import partial
import numpy as np
from astropy.modeling.models import Gaussian1D

from ..convolution import convolve, resample, gaussian_constant_r, boxcar


def test_resample_simple():
    # Test resampling of a constant spectrum
    w = np.linspace(500, 600, 1000)
    w2 = np.linspace(510, 590, 100)
    y = np.ones_like(w)
    y2 = resample(w2, w, y)
    assert np.allclose(y2, np.ones_like(y2))


def test_resample2():
    # Resample a Gaussian every other pixel
    w = np.linspace(500, 600, 1001)
    w2 = w[1:-1:2]
    y = np.exp(-0.5 * ((w - w[500]) / 4) ** 2)
    y2 = resample(w2, w, y)
    assert np.allclose(y2, y[1:-1:2], atol=2e-4)


def test_resample_log_wavelength():
    # Resample a Gaussian to a logarithmic wavelength scale
    w = np.linspace(500, 600, 1000)
    w2 = np.logspace(*np.log10([510, 590]), 500)
    gauss = Gaussian1D(mean=550, stddev=3)
    y = gauss(w)
    y2 = resample(w2, w, y)
    y3 = gauss(w2)
    assert np.allclose(y2, y3, atol=2e-3)


def test_convolve_boxcar():
    # Test convolution with a boxcar
    w = np.linspace(500, 600, 1001)
    y = np.zeros_like(w)
    y[500] = 1
    y2 = convolve(w, y, partial(boxcar, width=2), dw=1.5)
    assert np.allclose(y2[:490], 0)
    assert np.allclose(y2[490:511], 1/21)
    assert np.allclose(y2[511:], 0)


def test_convolve_gaussian():
    # Test convolution with a Gaussian
    w = np.linspace(500, 600, 1001)
    y = np.zeros_like(w)
    y[500] = 1
    y2 = convolve(w, y, partial(gaussian_constant_r, r=200), dw=9)
    sigma = 0.424661 * w[500] / 200
    y3 = Gaussian1D(amplitude=1 / (25.0663 * sigma), mean=w[500], stddev=sigma)(w)
    assert np.allclose(y2, y3)
