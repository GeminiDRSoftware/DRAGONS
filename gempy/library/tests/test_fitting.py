import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_less, assert_equal

from astropy.io import fits
from astropy.modeling.models import BlackBody, Gaussian1D, Gaussian2D
from astropy.utils import NumpyRNGContext
from astropy import units as u

from gempy.library.fitting import fit_1D

_RANDOM_SEED = 42
debug = False


# Simulate an object spectrum. This black body spectrum is not particularly
# meaningful physically; it's just a way to produce a credible continuum-like
# curve using something other than the functions being fitted:
cont_model = BlackBody(temperature=9500.*u.K, scale=5.e6)

# Some sky-line-like features to be rejected / fitted:
sky_model = (Gaussian1D(amplitude=2000., mean=5577., stddev=50.) +
             Gaussian1D(amplitude=1000., mean=6300., stddev=50.) +
             Gaussian1D(amplitude=300., mean=7914., stddev=50.) +
             Gaussian1D(amplitude=280., mean=8345., stddev=50.) +
             Gaussian1D(amplitude=310., mean=8827., stddev=50.))


class TestFit1D:
    """
    Some tests for gempy.library.fitting.fit_1D with 1-2D data.
    """
    def setup_class(self):

        # Co-ordinate grid in wavelength & pixels along the slit:
        wav = np.arange(3000, 10000, 50) * u.AA
        slit = np.arange(30)
        wav, slit = np.meshgrid(wav, slit)

        obj = (cont_model(wav).value *
               Gaussian1D(amplitude=1., mean=15.8, stddev=2.)(slit))

        # A continuum level makes for a more stable comparison of fit vs data:
        self.bglev = 30.
        sky = sky_model(wav.value) + self.bglev

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
        tolerances that roughly allow for model noise & fitting systematics
        and check that the expected pixels are rejected.
        """

        fit1d = fit_1D(self.data, weights=self.weights, function='chebyshev',
                       order=1, axis=0, sigma_lower=2.5, sigma_upper=2.5,
                       niter=5, plot=debug)
        fit_vals = fit1d.evaluate()

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

        # Also check that the rejected pixels were as expected (from previous
        # runs) for the central column:
        assert_equal(fit1d.mask[:12, 70], False)
        assert_equal(fit1d.mask[12:21, 70], True)
        assert_equal(fit1d.mask[21:, 70], False)

    def test_chebyshev_ax0_lin_grow2(self):
        """
        Fit background along the slit, rejecting the object spectrum with
        grow=2, which for these parameters produces a slightly closer match
        than grow=0.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=1, axis=0,
                          sigma_lower=3., sigma_upper=2.3, niter=5, grow=2,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.sky, atol=15., rtol=0.015)

    def test_chebyshev_def_ax_quartic(self):
        """
        Fit object spectrum with Chebyshev polynomial, rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=4,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=15., rtol=0.015)

    def test_chebyshev_ax1_quartic_grow2(self):
        """
        Fit object spectrum using higher thresholds than the last test to
        reject sky lines and grow=2 to compensate. Specify axis=1 explicitly,
        rather than the default of -1 (last axis).
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=4, axis=1,
                          sigma_lower=3.7, sigma_upper=3.7, niter=5, grow=2,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=15., rtol=0.015)

    def test_chebyshev_single_quartic(self):
        """
        Fit object spectrum, rejecting the sky in a single 1D array.
        """

        fit_vals = fit_1D(self.data[16], weights=self.weights[16],
                          function='chebyshev', order=4,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj[16] + self.bglev, atol=5.,
                        rtol=0.015)

    def test_chebyshev_1_model_def_ax_quartic(self):
        """
        Fit object spectrum in a single 1xN row, rejecting the sky.
        """

        fit_vals = fit_1D(self.data[16:17], weights=self.weights[16:17],
                          function='chebyshev', order=4,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj[16:17] + self.bglev, atol=5.,
                        rtol=0.015)

    def test_chebyshev_1_model_ax0_lin(self):
        """
        Fit linear sky background along a single Nx1 column, rejecting the
        object spectrum.
        """

        fit_vals = fit_1D(self.data[:, 70:71], weights=self.weights[:, 70:71],
                          function='chebyshev', order=1, axis=0,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.sky[:, 70:71], atol=10., rtol=0.)

    def test_chebyshev_ax0_lin_regions_noiter(self):
        """
        Fit linear sky background along the slit with the object spectrum
        region excluded by the user and no other rejection.
        """

        fit1d = fit_1D(self.data, weights=self.weights, function='chebyshev',
                       order=1, axis=0, niter=0, regions="1:10,23:30",
                       plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, self.sky, atol=20., rtol=0.02)

        assert fit1d.regions_pix == ((1,10),(23,30))

    def test_chebyshev_ax0_lin_slices_noiter(self):
        """
        Fit linear sky background along the slit with the object spectrum
        region excluded by the user and no other rejection (same test as above
        but passing a list of slice objects rather than a regions string).
        """
        fit1d = fit_1D(self.data, weights=self.weights, function='chebyshev',
                       order=1, axis=0, niter=0,
                       regions=[slice(0, 10), slice(22, 30)], plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, self.sky, atol=20., rtol=0.02)

        assert fit1d.regions_pix == ((1,10),(23,30))

    def test_lin_spline_ax0_ord1(self):
        """
        Fit object spectrum with a low-order cubic spline, rejecting the sky.
        Check that the pixel rejection is as expected, as well as the values.
        """

        fit1d = fit_1D(self.data, weights=self.weights, function='spline1',
                       order=1, axis=0, sigma_lower=2.5, sigma_upper=2.5,
                       niter=5, plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, self.sky, atol=20., rtol=0.02)

        # Also check that the rejected pixels were as expected (from previous
        # runs) for the central column:
        assert_equal(fit1d.mask[:11, 70], False)
        assert_equal(fit1d.mask[11:21, 70], True)
        assert_equal(fit1d.mask[21:, 70], False)

        assert fit1d.regions_pix == ((1,30),)

    def test_cubic_spline_def_ax_ord3(self):
        """
        Fit object spectrum with a low-order cubic spline, rejecting the sky
        with grow=1.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='spline3', order=3,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5, grow=1,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=20., rtol=0.01)

    def test_legendre_ax1_quartic(self):
        """
        Fit object spectrum with Legendre polynomial, rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='legendre', order=4, axis=1,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=20., rtol=0.01)

    def test_chebyshev_def_ax_quartic_masked(self):
        """
        Fit masked object spectrum with Chebyshev polynomial, rejecting sky.
        """

        fit1d = fit_1D(self.masked_data, weights=self.weights,
                       function='chebyshev', order=4,
                       sigma_lower=2.5, sigma_upper=2.5, niter=5, plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=20., rtol=0.01)

        # Ensure that masked input values have been passed through to the
        # output mask by the fitter:
        assert_equal(fit1d.mask[4:6,80:93], True)
        assert_equal(fit1d.mask[24:27,24:27], True)

        assert fit1d.regions_pix == ((1,140),)

    def test_cubic_spline_def_ax_ord3_masked(self):
        """
        Fit masked object spectrum with a low-order cubic spline, rejecting
        the sky with grow=1.
        """

        fit1d = fit_1D(self.masked_data, weights=self.weights,
                       function='spline3', order=3,
                       sigma_lower=2.5, sigma_upper=2.5, niter=5, grow=1,
                       plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=20., rtol=0.01)

        # Ensure that masked input values have been passed through to the
        # output mask by the fitter:
        assert_equal(fit1d.mask[4:6,80:93], True)
        assert_equal(fit1d.mask[24:27,24:27], True)

    def test_chebyshev_single_masked(self):
        """
        Fit a completely-masked single 1D array (which should return zeroes
        without actually fitting a model).
        """

        masked_row = np.ma.masked_array(self.data[16], mask=True)

        fit1d = fit_1D(masked_row, weights=self.weights[16],
                       function='chebyshev', order=4,
                       sigma_lower=2.5, sigma_upper=2.5, niter=5, plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, 0., atol=1e-6, rtol=0.)
        assert_equal(fit1d.mask, True)

    def test_chebyshev_1_unmasked_row(self):
        """
        Fit object spectrum with Chebyshev polynomial, where all rows but 1
        are masked (leaving a single model to fit, rather than a set).
        """

        masked_data = np.ma.masked_array(self.data, mask=False)
        masked_data.mask[1:] = True

        fit1d = fit_1D(masked_data, weights=self.weights, function='chebyshev',
                       order=4, sigma_lower=2.5, sigma_upper=2.5, niter=5,
                       plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals[0], (self.obj + self.bglev)[0],
                        atol=5., rtol=0.005)
        assert_allclose(fit_vals[1:], 0., atol=1e-6, rtol=0.)
        assert_equal(fit1d.mask[1:], True)

    def test_chebyshev_all_rows_masked(self):
        """
        Fit a completely-masked 2D array (which should return zeroes without
        actually fitting a model).
        """

        masked_data = np.ma.masked_array(self.data, mask=True)

        fit1d = fit_1D(masked_data, weights=self.weights, function='chebyshev',
                       order=4, sigma_lower=2.5, sigma_upper=2.5, niter=5,
                       plot=debug)
        fit_vals = fit1d.evaluate()

        assert_allclose(fit_vals, 0., atol=1e-6, rtol=0.)
        assert_equal(fit1d.mask, True)


class TestFit1DCube:
    """
    Some tests for gempy.library.fitting.fit_1D with 3D data.
    """
    def setup_class(self):

        # Co-ordinate grid in x, y & wavelength:
        wav = np.arange(3000, 10000, 50)
        x = np.arange(10)
        wav, y, x = np.meshgrid(wav, x, x, indexing='ij')

        obj = (cont_model(wav * u.AA).value *
               Gaussian2D(amplitude=1., x_mean=4.7, y_mean=5.2,
                          x_stddev=2.1, y_stddev=1.9)(x, y))

        # A continuum level makes for a more stable comparison of fit vs data:
        self.bglev = 30.
        sky = sky_model(wav) + self.bglev

        data = obj + sky

        # Add some noise:
        std = np.sqrt(36. + data)
        with NumpyRNGContext(_RANDOM_SEED):
            data += np.random.normal(0., 1., size=data.shape) * std

        self.obj, self.sky, self.data = obj, sky, data
        self.weights = 1. / std

    def test_chebyshev_ax0_quartic(self):
        """
        Fit object spectrum in x-y-lambda cube with Chebyshev polynomial,
        rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='chebyshev', order=4, axis=0,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=20., rtol=0.01)

    def test_chebyshev_def_ax_quartic(self):
        """
        Fit object spectrum in transposed lambda-y-x cube with Chebyshev
        polynomial, rejecting the sky.
        """

        # Here we transpose the input cube before fitting object spectra along
        # the default last axis, just because that makes more sense than trying
        # to fit the background with rejection along one spatial axis that is
        # too short to have clean sky regions.

        fit_vals = fit_1D(self.data.T, weights=self.weights.T,
                          function='chebyshev', order=4,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj.T + self.bglev, atol=20., rtol=0.01)

    def test_cubic_spline_ax0_ord3(self):
        """
        Fit object spectrum in x-y-lambda cube with cubic spline,
        rejecting the sky.
        """

        fit_vals = fit_1D(self.data, weights=self.weights,
                          function='spline3', order=3, axis=0,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj + self.bglev, atol=20., rtol=0.01)

    def test_cubic_spline_def_ax_ord3(self):
        """
        Fit object spectrum in transposed lambda-y-x cube with cubic spline,
        rejecting the sky.
        """

        fit_vals = fit_1D(self.data.T, weights=self.weights.T,
                          function='spline3', order=3,
                          sigma_lower=2.5, sigma_upper=2.5, niter=5,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, self.obj.T + self.bglev, atol=20., rtol=0.01)

    def test_cubic_spline_ax1_ord3_grow1(self):
        """
        Fit object spectrum in transposed x-lambda-y cube with cubic
        spline, rejecting the sky with grow=1.
        """

        fit_vals = fit_1D(np.rollaxis(self.data, 0, 2),
                          weights=np.rollaxis(self.weights, 0, 2),
                          function='spline3', order=3, axis=1,
                          sigma_lower=3.5, sigma_upper=3.5, niter=5, grow=1,
                          plot=debug).evaluate()

        assert_allclose(fit_vals, np.rollaxis(self.obj, 0, 2) + self.bglev,
                        atol=20., rtol=0.01)


class TestFit1DNewPoints:
    """
    Some tests for gempy.library.fitting.fit_1D evaluation at user-specified
    points instead of the original input points.
    """
    def setup_class(self):

        # A smooth function that we can try to fit:
        gauss = Gaussian1D(amplitude=10., mean=15.8, stddev=2.)

        # 1x and 5x sampled model values:
        self.data_coarse = np.tile(gauss(np.arange(30)) + 1, (3,1))
        self.data_fine = np.tile(gauss(np.arange(0, 30, 0.2)) + 1, (3,1))

    def test_chebyshev(self):

        # Fit the more-coarsely-sampled Gaussian model:
        fit1d = fit_1D(self.data_coarse, function='chebyshev', order=17,
                       niter=0, plot=debug)

        # Evaluate the fits onto 5x finer sampling:
        fit_vals = fit1d.evaluate(
            np.arange(0., self.data_coarse.shape[-1], 0.2)
        )

        # Compare fit values with the 5x sampled version of the original model
        # (ignoring the equivalent of the end 3 pixels from the input, where
        # the fit diverges a bit):
        assert_allclose(fit_vals[:,15:-15], self.data_fine[:,15:-15], atol=0.1)

    def test_spline(self):

        # Fit the more-coarsely-sampled Gaussian model:
        fit1d = fit_1D(self.data_coarse, function='spline3', order=15, niter=0,
                       plot=debug)

        # Evaluate the fits onto 5x finer sampling:
        fit_vals = fit1d.evaluate(
            np.arange(0., self.data_coarse.shape[-1], 0.2)
        )

        # Compare fit values with the 5x sampled version of the original model
        # (ignoring the equivalent of the end 3 pixels from the input, where
        # the fit diverges a bit):
        assert_allclose(fit_vals[:,15:-15], self.data_fine[:,15:-15], atol=0.1)
