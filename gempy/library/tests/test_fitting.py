import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less

from astropy.io import fits
from astropy.modeling.models import BlackBody, Gaussian1D
from astropy.utils import NumpyRNGContext
from astropy import units as u

from gempy.library.fitting import fit_1D

_RANDOM_SEED = 42
debug = False


class TestFit1D:
    """
    Some tests for gempy.library.fitting.fit_1D with 1-2D data.
    """
    def setup_class(self):

        # Co-ordinate grid in wavelength & pixels along the slit:
        wav = np.arange(3000, 10000, 50) * u.AA
        slit = np.arange(30)
        wav, slit = np.meshgrid(wav, slit)

        # Simulate an object spectrum. This black body spectrum is not
        # particularly meaningful physically; it's just a way to produce a
        # credible continuum-like curve using something other than the
        # functions being fitted:
        obj = (BlackBody(temperature=9500.*u.K, scale=5.e6)(wav).value *
               Gaussian1D(amplitude=1., mean=15.8, stddev=2.)(slit))

        # Some sky-line-like features to be rejected / fitted:
        sky = (Gaussian1D(amplitude=2000., mean=5577., stddev=50.) +
               Gaussian1D(amplitude=1000., mean=6300., stddev=50.) +
               Gaussian1D(amplitude=300., mean=7914., stddev=50.) +
               Gaussian1D(amplitude=280., mean=8345., stddev=50.) +
               Gaussian1D(amplitude=310., mean=8827., stddev=50.))(wav.value)
        # A continuum level makes for a more stable comparison of fit vs data:
        sky += 30.

        data = obj + sky

        # Add some noise:
        std = np.sqrt(36. + data)
        with NumpyRNGContext(_RANDOM_SEED):
            data += np.random.normal(0., 1., size=data.shape) * std

        # Make a copy that also has some bad pixels masked; the first masked
        # region is too long to get rejected automatically and test comparisons
        # will fail unless it is masked correctly:
        masked_data = np.ma.masked_array(data, mask=False, copy=True)
        badpix = np.ma.masked_array(1000., mask=True)
        masked_data[4:6, 80:93] = badpix
        masked_data[24:27, 24:27] = badpix

        self.obj, self.sky, self.data, self.std = obj, sky, data, std
        self.masked_data = masked_data
        self.weights = 1. / std
        # fits.writeto('testsim.fits', data)

    def test_chebyshev_ax0_lin(self):
        """
        Fit linear sky background along the slit, rejecting the object
        spectrum. Require resulting fit to match the true sky model within
        tolerances that roughly allow for model noise & fitting systematics.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=2, axis=0,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        # diff = abs(fit_vals - self.sky)
        # tol = 20 + 0.015 * abs(self.sky)
        # fits.writeto('diff.fits', diff)
        # fits.writeto('std.fits', 1.5*self.std)
        # fits.writeto('tol.fits', tol)
        # fits.writeto('where.fits', (diff > tol).astype(np.int16))

        # Stick to numpy.testing for comparing results, even if it gets a bit
        # convoluted, because it performs extra checks and reports useful
        # information in case of failure. All of the following comparison
        # methods work, but I'm still a bit undecided on the optimal criterion
        # for success/failure:
        assert_allclose(fit_vals, self.sky, atol=20., rtol=0.015)
        # assert not np.any(np.abs(fit_vals-self.sky) > 2.*self.std)
        # assert_array_less(np.abs(fit_vals-self.sky), 2.*self.std)

    def test_chebyshev_ax0_lin_grow2(self):
        """
        Fit background along the slit, rejecting the object spectrum with
        grow=2, which for these parameters produces a slightly closer match
        than grow=0.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=2, axis=0,
                          lsigma=3., hsigma=2.3, iterations=5, grow=2,
                          plot=debug)

        assert_allclose(fit_vals, self.sky, atol=15., rtol=0.015)

    def test_chebyshev_def_ax_quintic(self):
        """
        Fit object spectrum with Chebyshev polynomial, rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=5,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        assert_allclose(fit_vals, self.obj, atol=40., rtol=0.025)

    def test_chebyshev_ax1_quintic_grow2(self):
        """
        Fit object spectrum using higher thresholds than the last test to
        reject sky lines and grow=2 to compensate. Specify axis=1 explicitly,
        rather than the default of -1 (last axis).
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=5, axis=1,
                          lsigma=3.7, hsigma=3.7, iterations=5, grow=2,
                          plot=debug)

        assert_allclose(fit_vals, self.obj, atol=40., rtol=0.02)

    def test_chebyshev_single_quintic(self):
        """
        Fit object spectrum, rejecting the sky in a single 1D array.
        """

        fit_vals = fit_1D(self.data[16], weights=self.weights[16],
                          function='chebyshev', order=5,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        assert_allclose(fit_vals, self.obj[16], atol=30., rtol=0.02)

    def test_chebyshev_1_model_def_ax_quintic(self):
        """
        Fit object spectrum in a single 1xN row, rejecting the sky.
        """

        fit_vals = fit_1D(self.data[16:17], weights=self.weights[16:17],
                          function='chebyshev', order=5,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        # This should work, but currently fails because fit_1D is returning
        # a result with shape (140, 1) from (1, 140) inputs.

        assert_allclose(fit_vals, self.obj[16:17], atol=30., rtol=0.02)

    def test_chebyshev_1_model_ax0_lin(self):
        """
        Fit linear sky background along a single Nx1 column, rejecting the
        object spectrum.
        """

        fit_vals = fit_1D(self.data[:, 70:71], weights=self.weights[:, 70:71],
                          function='chebyshev', order=2, axis=0,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        assert_allclose(fit_vals, self.sky[:, 70:71], atol=10., rtol=0.)

    def test_chebyshev_ax0_lin_regions_noiter(self):
        """
        Fit linear sky background along the slit with the object spectrum
        region excluded by the user and no other rejection.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=2, axis=0, iterations=0,
                          regions="1:10,23:30", plot=debug)

        assert_allclose(fit_vals, self.sky, atol=20., rtol=0.02)

    def test_lin_spline_ax0_ord1(self):
        """
        Fit object spectrum with a low-order cubic spline, rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='spline1', order=1, axis=0,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        assert_allclose(fit_vals, self.sky, atol=20., rtol=0.02)

    def test_cubic_spline_def_ax_ord3(self):
        """
        Fit object spectrum with a low-order cubic spline, rejecting the sky
        with grow=1.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='spline3', order=3,
                          lsigma=2.5, hsigma=2.5, iterations=5, grow=1,
                          plot=debug)

        assert_allclose(fit_vals, self.obj, atol=40., rtol=0.02)

    def test_legendre_ax1_quintic(self):
        """
        Fit object spectrum with Legendre polynomial, rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='legendre', order=5, axis=1,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        assert_allclose(fit_vals, self.obj, atol=40., rtol=0.02)

    def test_chebyshev_def_ax_quintic_masked(self):
        """
        Fit masked object spectrum with Chebyshev polynomial, rejecting sky.
        """

        fit_vals = fit_1D(self.masked_data, weights=self.weights,
                          function='chebyshev', order=5,
                          lsigma=2.5, hsigma=2.5, iterations=5,
                          plot=debug)

        assert_allclose(fit_vals, self.obj, atol=40., rtol=0.02)

    def test_cubic_spline_def_ax_ord3_masked(self):
        """
        Fit masked object spectrum with a low-order cubic spline, rejecting
        the sky with grow=1.
        """

        fit_vals = fit_1D(self.masked_data, weights=self.weights,
                          function='spline3', order=3,
                          lsigma=2.5, hsigma=2.5, iterations=5, grow=1,
                          plot=debug)

        assert_allclose(fit_vals, self.obj, atol=40., rtol=0.02)

