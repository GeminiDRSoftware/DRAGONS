#!/usr/bin/env python
"""
Tests for the :mod:`~gempy.library.tracing` module.
"""

import numpy as np
import pytest

from astropy.modeling import models
from gempy.library import tracing


@pytest.mark.parametrize("x0, fwhm", [(1600, 2), (1600, 4), (1600, 7)])
def test_estimate_peak_width_with_one_line(x0, fwhm):

    stddev_to_fwhm = 2 * np.sqrt(2 * np.log(2))
    stddev = fwhm / stddev_to_fwhm

    x = np.arange(0, 3200)
    g = models.Gaussian1D(mean=x0, stddev=stddev)
    y = g(x)

    measured_fwhm = tracing.estimate_peak_width(y)

    np.testing.assert_allclose(fwhm, measured_fwhm, rtol=0.10)


@pytest.mark.parametrize("noise", np.arange(0, 0.5, 0.1))
def test_estimate_peak_width_with_one_line_with_noise(noise):

    x0 = 1600
    fwhm = 4

    stddev_to_fwhm = 2 * np.sqrt(2 * np.log(2))
    stddev = fwhm / stddev_to_fwhm

    x = np.arange(0, 3200)
    g = models.Gaussian1D(mean=x0, stddev=stddev)

    np.random.seed(0)
    y = g(x) + noise * (np.random.rand(x.size) - 0.5)

    measured_fwhm = tracing.estimate_peak_width(y)

    np.testing.assert_allclose(fwhm, measured_fwhm, rtol=0.10 + 0.4 * noise)


def test_find_peaks():
    x = np.arange(0, 3200)
    y = np.zeros_like(x, dtype=float)
    n_peaks = 20

    stddev = 8.
    peaks = np.linspace(
        x.min() + 0.05 * x.ptp(), x.max() - 0.05 * x.ptp(), n_peaks)

    for x0 in peaks:
        g = models.Gaussian1D(mean=x0, stddev=stddev)
        y += g(x)

    peaks_detected, _ = tracing.find_peaks(y, np.ones_like(y) * stddev)

    np.testing.assert_allclose(peaks_detected, peaks, atol=0.5)


def test_find_peaks_raises_typeerror_if_mask_is_wrong_type():
    x = np.arange(0, 3200)

    stddev = 8
    x0 = x.min() + 0.5 * x.ptp()
    g = models.Gaussian1D(mean=x0, stddev=stddev)
    y = g(x)

    mask = np.zeros_like(y)

    with pytest.raises(TypeError):
        peaks_detected, _ = tracing.find_peaks(y, np.ones_like(y) * stddev, mask=mask)

