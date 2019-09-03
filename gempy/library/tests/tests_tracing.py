#!/usr/bin/env python
"""
Tests for the :mod:`~gempy.library.tracing` module.
"""

import numpy as np
import pytest

from astropy.modeling import models
from gempy.library import tracing


@pytest.mark.xfail(reason='Check if test is correct')
@pytest.mark.parametrize("x0, stddev", [(1600, 8)])
def test_estimate_peak_width_with_one_line(x0, stddev):
    x = np.arange(0, 3200)
    g = models.Gaussian1D(mean=x0, stddev=stddev)
    y = g(x)

    stddev_to_fwhm = 2 * np.sqrt(2 * np.log(2))
    measured_width = tracing.estimate_peak_width(y)
    measured_stddev = measured_width / stddev_to_fwhm

    assert np.testing.assert_approx_equal(measured_stddev, stddev)

