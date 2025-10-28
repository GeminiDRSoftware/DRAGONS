#!/usr/bin/env python
"""
Tests for the :mod:`~gempy.library.tracing` module.
"""

import numpy as np
import pytest

from astropy.modeling import models
from gempy.library import peak_finding


@pytest.mark.parametrize("fwhm", [2, 4, 6, 8, 12])
def test_estimate_peak_width(fwhm):
    n_peaks = 10
    noise = 0.001

    x = np.arange(0, 1000)
    y = np.zeros_like(x, dtype=float)

    stddev_to_fwhm = 2 * np.sqrt(2 * np.log(2))
    stddev = fwhm / stddev_to_fwhm

    peaks = np.linspace(
        x.min() + 0.05 * np.ptp(x), x.max() - 0.05 * np.ptp(x), n_peaks)

    for x0 in peaks:
        g = models.Gaussian1D(mean=x0, stddev=stddev)
        y += g(x)

    np.random.seed(0)
    y += (np.random.random(x.size) - 0.5) * noise

    measured_fwhm = peak_finding.estimate_peak_width(y)

    np.testing.assert_allclose(fwhm, measured_fwhm, atol=1)


@pytest.mark.parametrize("noise", [0.01, 0.1, 0.2, 0.4])
@pytest.mark.skip("Test is failing and need to be checked")
def test_find_peaks(noise):

    x = np.arange(0, 3200)
    y = np.zeros_like(x, dtype=float)
    n_peaks = 20

    stddev = 4.
    peaks = np.linspace(
        x.min() + 0.05 * np.ptp(x), x.max() - 0.05 * np.ptp(x), n_peaks)

    for x0 in peaks:
        g = models.Gaussian1D(mean=x0, stddev=stddev, amplitude=100)
        y += g(x)

    np.random.seed(0)
    y += (np.random.random(x.size) - 0.5) * noise

    peaks_detected, _, _ = peak_finding.find_wavelet_peaks(y, np.ones_like(y) * stddev)

    np.testing.assert_allclose(peaks_detected, peaks, atol=1)


def test_get_limits():
    CENT, SIG = 250, 20
    x = np.arange(CENT * 2)
    y = 100 * np.exp(-0.5 * ((x - CENT) / SIG) ** 2)

    limits = np.asarray(peak_finding.get_limits(y, mask=None, peaks=[CENT],
                                           threshold=0.05)[0])
    np.testing.assert_allclose(abs(limits - CENT),
                               SIG * np.sqrt(2 * np.log(20)),
                               atol=0.1*SIG)
