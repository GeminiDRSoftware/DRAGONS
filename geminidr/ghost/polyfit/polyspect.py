"""
Common tools for a generic spectrograph.

This code was originally written as a simple simulation/extraction
definition code for the RHEA instrument.

..
    ORIGINAL DOCSTRING
    This is a simple simulation and extraction definition code for RHEA.
    The key is to use Tobias's spectral fitting parameters and define
    the same kinds of extraction arrays as pyghost.
"""
from __future__ import division, print_function

import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
import scipy.optimize as op
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from astropy.modeling import Fittable2DModel, Parameter

# pylint: disable=maybe-no-member, too-many-instance-attributes

class Polyspect(object):
    """
    A class containing tools common for any spectrograph.

    This class should be inherited by the specific spectrograph module.
    Contains functions related to polynomial modelling of spectrograph orders.

    The class initialisation takes a series of inputs that define the
    spectrograph orders and orientation. Most of the parameters are
    self-explanatory, but here is a list of those that may not be:

    +------------------------------+-------------------------------------------+
    | **Variable Name**            | **Purpose and meaning**                   |
    +------------------------------+-------------------------------------------+
    | ``m_ref``, ``m_min`` and     | Reference, minimum and maximum order      |
    | ``m_max``                    | indices for the camera.                   |
    +------------------------------+-------------------------------------------+
    | ``szx`` and ``szy``          | number of pixels in the x and y directions|
    +------------------------------+-------------------------------------------+
    | ``transpose``                | Whether the CCD orientation is transposed,|
    |                              | e.g. if the spectral direction is along y.|
    +------------------------------+-------------------------------------------+
    | ``x_map`` and                | The spatial and wavelength maps of the    |
    | ``w_map``                    | CCD. These are populated by               |
    |                              | :any:`spectral_format_with_matrix`        |
    +------------------------------+-------------------------------------------+
    | ``blaze`` and                | The blaze function and rotation matrices. |
    | ``matrices``                 | These are populated by                    |
    |                              | :any:`spectral_format_with_matrix`        |
    +------------------------------+-------------------------------------------+

    Attributes
    ----------
    m_ref: int
        The reference order, typically whatever order number is the middle
        of the range of orders on the CCD
    szx: int
        The number of CCD pixels in the x direction
    szy: int
        The number of CCD pixels in the y direction
    m_min: int
        The lowest order number
    m_max: int
        The highest order number
    transpose: bool
        Boolean; whether the CCD is transposed relative to x in the
        spectral direction and y in the spatial.
    x_map: :obj:`numpy.ndarray`
        This is the [n_orders x spectral_pixels] array containing the x
        value along each order
    w_map: :obj:`numpy.ndarray`
        Wavelength scale map. Same shape as x_map
    blaze: :obj:`numpy.ndarray`
        Blaze angle map. Same shape as x_map
    matrices: :obj:`numpy.ndarray`
        Rotation matrices as a function of pixel in the spectral direction
        for all orders.
    """

    def __init__(self, m_ref, szx, szy, m_min, m_max, transpose):
        # All necessary parameters are listed here and initialized by the
        # parent class
        self.m_ref = m_ref
        self.szx = szx
        self.szy = szy
        self.m_min = m_min
        self.m_max = m_max
        # True if the spectral dispersion dimension is over the x (column) axis
        self.transpose = transpose
        self.x_map = None
        self.w_map = None
        self.blaze = None
        self.matrices = None

    def evaluate_poly(self, params, data=None):
        """
        Evaluates a polynomial of polynomials, given model parameters.

        This function takes a set of polynomial coefficients and
        returns the evaluated polynomials in all spatial pixels
        for all orders.

        This function is the key function that enables polynomial
        descriptions of the various aspects of the spectrograph to work. In its
        simplest form, this function converts the polynomial coefficients into
        values as a function of y pixel and spectrograph order. The optional
        ``data`` input is only provided if the desired structure of the output
        is not the default (orders, szy) evaluation at all points.

        This function is designed such that any set of polynomial coefficients
        can be given and the evaluation will take place.

        See Also
        --------
        :meth:`spectral_format`
        :meth:`fit_resid`

        Parameters
        ----------

        params: :obj:`numpy.ndarray`
            Model parameters with the coefficients to evaluate.

        data: :obj:`list` (optional)
            Optional data input for the y_values and orders. This dictates
            an alternative format for the returned function evaluation. The
            default is a [n_orders x spectral_pixel] float array.

        Raises
        ------
        TypeError
            If required input ``params`` is not provided.
        
        Returns
        -------

        evaluation: :obj:`numpy.ndarray`
            This is a (orders,yvalues) array containing the polynomial
            evaluation at every point. If data is provided, the returned
            array has the same shape. 

        """
        # params needs to be a np.array
        if not isinstance(params, np.ndarray):
            raise TypeError('Please provide params as a numpy float array')
        # The polynomial degree as a function of y position.
        ydeg = params.shape[0] - 1

        if data is not None:
            y_values, orders = data
        else:
            # Create the y_values and orders.
            # This is essentially the purpose of creating this function. These
            # parameters can be easily derived from class properties and should
            # not have to be provided as inputs. 
            y_values, orders = np.meshgrid(np.arange(self.szy),
                                           np.arange(self.m_max -
                                                     self.m_min + 1) +
                                           self.m_min)
        # However, we should just use the orders as a single array.
        if orders.ndim > 1:
            orders = orders[:, 0]
        mprime = float(self.m_ref) / orders - 1
        # In case of a single polynomial, this solves the index problem.
        if params.ndim == 1:
            polyp = np.poly1d(params)
            evaluation = np.meshgrid(np.arange(self.szy), polyp(mprime))[1]
        else:
            # Initiate a polynomials array.
            polynomials = np.empty((len(orders), ydeg + 1))
            # Find the polynomial coefficients for each order.
            for i in range(ydeg + 1):
                polyq = np.poly1d(params[i, :])
                polynomials[:, i] = polyq(mprime)
            evaluation = np.empty(y_values.shape)
            # The evaluate as a function of position.
            for i in range(len(orders)):
                polyp = np.poly1d(polynomials[i, :])
                evaluation[i] = polyp(y_values[i] - self.szy // 2)
        return evaluation

    def fit_resid(self, params, orders, y_values, data, ydeg=3, xdeg=3,
                  sigma=None):
        """
        A fit function for :meth:`read_lines_and_fit` and :meth:`fit_to_x`.

        This function is to be used in :any:`scipy.optimize.leastsq` as the
        minimization function.
        The same function is used in :any:`fit_to_x`, but in that case
        "waves" is replaced by "xs".

        This function was initially designed for the wavelength fit only, but
        later generalised for both wavelength and order position fitting. The
        variable names are still named with the wavelength fitting mindset, but
        the principle is the same.

        Parameters
        ----------

        params: :obj:`numpy.ndarray` array
            2D array containing the polynomial coefficients that will form the
            model to be compared with the real data.
        orders: :obj:`array`, data type int
            The order numbers for the residual fit repeated ys times.
        y_values: :obj:`numpy.ndarray` array
            This is an orders x y sized array with pixel indices on y direction
            for each order.
        data: :obj:`numpy.ndarray` array
            This is the data to be fitted, which will have the model subtracted
            from
        ydeg: int
            Polynomial degree as a function of order
        xdeg: int
            Polynomial degree as a function of y
        sigma: :obj:`numpy.ndarray` array
            Array containing uncertainties for each point. Must have the same 
            format as data. 

        Returns
        -------
        :obj:`numpy.ndarray`
            The residual between the model and data supplied.
        """
        # params needs to be a np.array
        if not isinstance(params, np.ndarray):
            raise TypeError('Please provide params as a numpy float array')
        if not isinstance(orders, np.ndarray):
            raise TypeError('Please ensure orders is a numpy float array')
        if not isinstance(y_values, np.ndarray):
            raise TypeError('Please provide y_values as a numpy float array')
        if not isinstance(data, np.ndarray):
            raise TypeError('Please provide data as a numpy float array')

        params = params.reshape((ydeg + 1, xdeg + 1))
        if len(orders) != len(y_values):
            raise UserWarning("orders and y_values must all be the same "
                              "length!")

        result = self.evaluate_poly(params, (y_values, orders))
        if sigma is None:
            sigma = np.ones_like(result)

        return (data - result) / sigma

    def read_lines_and_fit(self, init_mod, arclines, ydeg=3, xdeg=3):
        """
        Fit to an array of spectral data using an initial model parameter set.

        This function takes an initial wavelength scale guess and a list of
        actual arc lines and their wavelengths, sets up the least squares
        minimisation and fits to the measured positions of the arc lines by the
        ``find_lines`` function in the :doc:`extract` module.

        The functional form of the polynomial of polynomials is:

        .. math::

            \\textrm{wave} = p_0(m) + p_1(m)\\times y^\\prime + p_2(m)\\times y'^{2} + ...\\textrm{,}

        with :math:`y^{\\prime} = y - y_{\\textrm{middle}}`, and:

        .. math::

            p_0(m) = q_{00} + q_{01} \\times m' + q_{02} \\times m'^2 + ...\\textrm{,}

        with :math:`m' = m_{\\rm ref}/m - 1`.

        | This means that the simplest spectrograph model should have:
        | :math:`q_{00}` : central wavelength of order :math:`m_\\textrm{ref}`
        | :math:`q_{01}` : central wavelength of order :math:`m_\\textrm{ref}`
        | :math:`q_{10}` : central wavelength :math:`/R_{\\textrm{pix}}`,
        | :math:`q_{11}` : central_wavelength :math:`/R_{\\textrm{pix}}`,
        | with :math:`R_\\textrm{pix}` the resolving power :math:`/` pixel.
        | All other model parameters will be (approximately) zero.

        Parameters
        ----------
        init_mod: :any:`array`-like, two-dimensional
            initial model parameters.
        arclines: :any:`array`-like
            wavelengths of lines from the :any:`find_lines` function.
        xdeg/ydeg: int
            Order of polynomial

        Returns
        -------
        params: :obj:`numpy.ndarray` array
            Fitted parameters
        wave_and_resid: :obj:`numpy.ndarray` array
            Wavelength and fit residuals.
        """
        # The next loop reads in wavelengths from a file.
        # To make this neater, it could be a function that overrides this
        # base class.
        # This code needs more work since we will only have one format
        # for a GHOST arc line list
        # FIXME Avoid hard-coding values for array positions
        lines = arclines
        orders = lines[:, 3]
        waves = lines[:, 0]
        y_values = lines[:, 1]
        ydeg = init_mod.shape[0] - 1
        xdeg = init_mod.shape[1] - 1
        # For weighted fitting purposes, use the maximum of the Gaussian fit.
        sigma = 1. / lines[:, 4]

        # Now we proceed to the least squares minimization.
        # We provide the fit_resid function as the minimization function
        # and the initial model. All required arguments are also provided.
        bestp = op.leastsq(self.fit_resid, init_mod, args=(orders, y_values,
                                                           waves, ydeg, xdeg))
                                                           #sigma))
        final_resid = self.fit_resid(bestp[0], orders, y_values, waves,
                                     ydeg=ydeg, xdeg=xdeg)
        # Output the fit residuals.
        wave_and_resid = np.array([waves, orders, final_resid]).T
        print("Fit residual RMS (Angstroms): {0:6.3f}".format(
            np.std(final_resid)))
        params = bestp[0].reshape((ydeg + 1, xdeg + 1))

        return params, wave_and_resid

    def spectral_format(self, wparams=None, xparams=None, img=None):
        """
        Form a spectrum from wavelength and polynomial models.

        .. note::

            This is a simpler version of :any:`spectral_format_with_matrix`
            (which replaces this) that does not take the slit rotation into
            consideration.

        Create a spectrum, with wavelengths sampled in 2 orders based on
        a pre-existing wavelength and x position polynomial model.
        This code takes the polynomial model and calculates the result as
        a function of order number scaled to the reference order and then
        as a function of y position.
        Optionally an image can be supplied for the model to be overlaid
        on top of.

        Parameters
        ----------
        wparams: :obj:`numpy.ndarray`, optional
            2D array with polynomial parameters for wavelength scale
        xparams: :obj:`numpy.ndarray`, optional
            2D array with polynomial parameters for x scale
        img: :obj:`numpy.ndarray`, optional
            2D array containing an image. This function
            uses this image and over plots the created position model.

        Raises
        ------
        UserWarning:
            All inputs are notionally optional but some combination is required.
            Therefore several checks are needed to ensure that a suitable
            combination of those is required for successful implementation
            of this function. This warning is raised if not enough inputs are
            provided or the wrong format is given.

        Returns
        -------
        x:  :obj:`numpy.ndarray` (nm, ny)
            The x-direction pixel co-ordinate corresponding to each y-pixel and
            each order (m).
        wave:  :obj:`numpy.ndarray` (nm, ny)
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: :obj:`numpy.ndarray` (nm, ny)
            The blaze function (pixel flux divided by order center flux)
            corresponding to each y-pixel and each order (m).
        ccd_centre: :obj:`dict`
            Parameters of the internal co-ordinate system describing the
            center of the CCD.

            .. warning::
                Not yet implemented.
        """

        # FIXME this function requires more detail in comments about procedure

        # Now lets interpolate onto a pixel grid rather than the arbitrary
        # wavelength grid we began with.
        norders = self.m_max - self.m_min + 1

        if (xparams is None) and (wparams is None):
            raise UserWarning(
                'Must provide at least one of xparams or wparams')
        if (xparams is not None) and (not isinstance(xparams, np.ndarray)):
            raise UserWarning('xparams provided with invalid format')
        if (wparams is not None) and (not isinstance(wparams, np.ndarray)):
            raise UserWarning('wparams provided with invalid format')

        # An initialisation of the y_values and order arrays
        y_values, orders = np.meshgrid(np.arange(self.szy),
                                       np.arange(self.m_max - self.m_min + 1) +
                                       self.m_min)
        order = orders[:, 0]
        if wparams is None:
            wparams = np.ones((3, 3))
        wave_int = self.evaluate_poly(wparams)
        x_int = self.evaluate_poly(xparams)

        # Finally, the blaze
        wcen = wave_int[:, int(self.szy / 2)]
        disp = wave_int[:, int(self.szy / 2 + 1)] - wcen

        order_width = (wcen / order) / disp
        order_width = np.meshgrid(np.arange(self.szy), order_width)[1]
        blaze_int = np.sinc((y_values - self.szy / 2)
                            / order_width) ** 2

        # Plot this if we have an image file
        if (img is not None) and (xparams is not None):
            if not isinstance(img, np.ndarray):
                raise UserWarning('img must be numpy array')
            if img.ndim != 2:
                raise UserWarning('Image array provided is not a 2 dimensional '
                                  'array')
            if not self.transpose:
                img = img.T
            plt.clf()
            plt.imshow(np.arcsinh(np.maximum(img - np.median(img),0) / 100), aspect='auto',
                       interpolation='nearest', cmap=cm.gray)
            plt.axis([0, img.shape[1], img.shape[0], 0])
            plt.plot(x_int.T + + self.szx // 2)

        return x_int, wave_int, blaze_int

    def adjust_x(self, old_x, image, num_xcorr=21):
        """
        Adjusts the x-pixel value mapping.

        Adjust the x-pixel value based on an image and an initial array from
        spectral_format(). This function performs a cross correlation between a
        pixel map and a given 2D array image and calculates a shift along the
        spatial direction. The result is used to inform an initial global shift
        for the fitting procedure. Only really designed for a single fiber flat,
        single fiber science image or a convolution map. This is a helper
        routine for :any:`fit_x_to_image`.

        This is an assistant function designed to slightly modify the initial
        position fitting for any global shift in the spatial direction. The
        usefulness of this function can be debated, but ultimately it won't
        create any problems.

        
        Parameters
        ----------
        old_x: :obj:`numpy.ndarray`
            An old x pixel array
        image: :obj:`numpy.ndarray`
            A 2D image array to be used as the basis for the adjustment.
        num_xcorr: int, optional
            Size of the cross correlation function. This should be an indication
            of how much the cross correlation should move.

        Returns
        -------
        new_x: :obj:`numpy.ndarray`
             A new adjusted value of the x array.
        """
        if not isinstance(old_x, np.ndarray):
            raise TypeError('old_x must be a numpy array')
        if not isinstance(image, np.ndarray):
            raise TypeError('image must be a numpy array')
        if image.ndim != 2:
            raise UserWarning('image array must be 2 dimensional')

        # Create an array with a single pixel with the value 1.0 at the
        # expected peak of each order.
        single_pix_orders = np.zeros(image.shape)
        xygrid = np.meshgrid(np.arange(old_x.shape[0]),
                             np.arange(old_x.shape[1]))
        single_pix_orders[np.round(xygrid[1]).astype(int),
                          np.round(old_x.T + self.szx // 2).astype(int)] = 1.0

        # Make an array of cross-correlation values.
        xcorr = np.zeros(num_xcorr)
        for i in range(num_xcorr):
            xcorr[i] = np.sum(np.roll(single_pix_orders, i - num_xcorr // 2,
                                      axis=1) * image)

        # Based on the maximum cross-correlation, adjust the model x values.
        the_shift = np.argmax(xcorr) - num_xcorr // 2
        new_x = old_x + the_shift
        return new_x

    def fit_x_to_image(self, data, xparams, decrease_dim=8, sampling=1,
                       search_pix=15, inspect=False):
        """
        Fit a "tramline" map.

        This is the main function that fits the "tramline" map (i.e. the order
        trace) to the image. The initial map has to be quite close and can be
        adjusted with the :any:`adjust_model` function, and then fitted further.

        This particular function sets up the data to be fitted, and then calls
        :any:`fit_to_x` to do the actual fit. The segmentation of the image into
        smaller sections is done here, to limit the number of parameters to fit
        and make the fitting faster. The default behaviour is to subdivide the
        image by a factor of 8 in the spectral direction to reduce the
        dimensions of the fit. This is acceptable, because the final fit *must*
        be smooth anyway.

        This function also is responsible for taking the image to fit, typically
        the result of the smoothed flat convolution, and finding the location of
        the maxima along the orders for fitting purposes. The process of
        cleaning up the flats is done in the main processing primitives, but
        here the weighting for each point is determined based on the peak flux.
        This weighting is done to ensure the fainter order edges are not
        contributing to the fit as much.

        Optionally, the ``inspect`` parameter will display the initial order
        centre maxima locations with point sizes following the weights, and then
        the result of the fit.

        Note that an initial map has to be pretty close, i.e. within
        ``search_pix`` everywhere. To get within ``search_pix`` everywhere, a
        simple model with a few parameters is fitted manually. This can be done
        with the GUI using the :any:`adjust_model` function and finally adjusted
        with the :any:`adjust_x` function.

        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            The image of a single reference fiber to fit to. Typically
            the result of the convolution.
        xparams: :obj:`numpy.ndarray`
            The polynomial parameters to be fitted.
        decrease_dim: int, optional
            Median filter by this amount in the dispersion direction and
            decrease the dimensionality of the problem accordingly.
            This helps with both speed and robustness.
        sampling: int
            Further speed up problem by regularly sampling datapoints in
            the dispersion direction
        search_pix: int, optional
            Search within this many pixels of the initial model.
        inspect: bool, optional
            If true, once fit is done the adjust_model function
            is called so that the user can inspect the result of the
            fit and decide if it is good enough.

        Raises
        ------
        UserWarning
            If the decreased dimension is not possible due to rounding off
            errors
        
        Returns
        -------
        fitted_parameters: :obj:`numpy.ndarray`
            The new model parameters fitted.

        """
        # FIXME more detailed comments, more consistent input checking

        xbase, wave, blaze = self.spectral_format(xparams=xparams)
        if self.transpose:
            image = data.T
        else:
            image = data

        # Now use the adjust function to figure out a global shift in
        # the spatial direction
        x_values = self.adjust_x(xbase, image)

        if image.shape[0] % decrease_dim != 0:
            raise UserWarning(
                "Can not decrease image dimension by this amount. "
                "Please check if the image size in the spectral dimension "
                "is exactly divisible by this amount.")
        # Median-filter in the dispersion direction.
        # This process will 'collapse' the image in the spectral direction
        # and make sure the fit is faster.
        image_med = image.reshape((image.shape[0] // decrease_dim,
                                   decrease_dim, image.shape[1]))
        image_med = np.mean(image_med, axis=1)
        order_y = np.meshgrid(np.arange(xbase.shape[1]), # pylint: disable=maybe-no-member
                              np.arange(xbase.shape[0]) + self.m_min)  # pylint: disable=maybe-no-member
        y_values = order_y[0]
        y_values = np.average(y_values.reshape(x_values.shape[0],
                                               x_values.shape[1] //
                                               decrease_dim,
                                               decrease_dim), axis=2)
        x_values = np.average(x_values.reshape(x_values.shape[0],
                                               x_values.shape[1] //
                                               decrease_dim,
                                               decrease_dim), axis=2)
        sigma = np.ones_like(x_values)

        # Now go through and find the peak pixel values.
        # Do this by searching for the maximum value along the
        # order for search_pix on either side of the initial
        # model pixels in the spatial direction.
        for j in range(x_values.shape[1]):
            xindices = np.round(x_values[:, j]).astype(int)
            for i, xind in enumerate(xindices):
                lpix = max(0, self.szx // 2 + xind - search_pix)
                rpix = min(self.szx // 2 + xind + search_pix + 1,
                           image_med.shape[1] - 1)
                peakpix = lpix + np.argmax(image_med[j, lpix:rpix])
                new_peak = 0.5 * ((image_med[j, peakpix+1] - image_med[j, peakpix-1]) /
                                  (3 * image_med[j, peakpix] - image_med[j, peakpix-1:peakpix+2].sum()))
                if abs(new_peak) < 1:
                    x_values[i, j] = new_peak + peakpix - self.szx // 2
                    sigma[i, j] = 1. / np.sqrt(image_med[j, peakpix])
                else:
                    sigma[i, j] = 1E5
                #if i == 12 and sigma[i, j] < 1e5:
                #    print(j, y_values[i, j], x_values[i, j] + self.szx // 2)
                #    print(image_med[j, peakpix-1:peakpix+2])

        #print("SHAPE", y_values.shape)
        #for xx, yy in zip(x_values.flatten(), y_values.flatten()):
        #    print(f"{yy:6.1f} {xx+self.szx//2:10.4f}")
        # The inspect flag is used if a display of the results is desired.
        if inspect:
            plt.ioff()
            plt.clf()
            plt.imshow(data, cmap="gray")
            point_sizes = 36*np.median(sigma)/sigma
            plt.scatter(y_values.T, x_values.T + self.szx // 2,
                        marker = '.',
                        s = point_sizes.T.flatten(),
                        color = 'red')
            plt.show()

        fitted_params = self.fit_to_x(x_values[:, ::sampling], xparams, y_values=y_values[:, ::sampling],
                                      sigma=sigma[:, ::sampling])
        if inspect:
            # This will plot the result of the fit once successful so
            # the user can inspect the result.
            plt.clf()
            plt.imshow((data - np.median(data)) / 1e2)
            x_int, wave_int, blaze_int = \
                self.spectral_format(wparams=None, xparams=fitted_params)
            ygrid = np.meshgrid(np.arange(data.shape[1]),
                                np.arange(x_int.shape[0]))[
                0]  # pylint: disable=maybe-no-member
            plt.plot(ygrid, x_int + data.shape[0] // 2,
                     color='green', linestyle='None', marker='.')
            plt.show()
            plt.ion()

        return fitted_params

    def fit_to_x(self, x_to_fit, init_mod, y_values=None, sigma=None,
                 decrease_dim=1, maxiter=5, sigma_rej=3):
        """
        Fit to an (norders, ny) array of x-values.

        This is the helper function that takes the setup from
        :any:`fit_x_to_image` and actually fits the x model. This function also
        has a `decrease_dim` dimension adjustment argument, but is not used by
        default.

        The functional form is:

        .. math::

            x = p_0(m) + p_1(m)\\times y' + p_2(m)\\times y'^2 + \\ldots\\textrm{,}

        with :math:`y' = y - y_{\\rm middle}`, and:

        .. math::

            p_0(m) = q_{00} + q_{01} \\times m' + q_{02} \\times m'^2 + \\ldots\\textrm{,}

        with :math:`m' = m_{\\rm ref}/m - 1`.

        | This means that the simplest spectrograph model should have:
        | :math:`q_{00}` : central order y pixel;
        | :math:`q_{01}`:  spacing between orders divided by the number of
          orders;
        | ...with everything else approximately zero.

        Parameters
        ----------
        x_to_fit: :obj:`numpy.ndarray`
            x values to fit. This should be an (orders,y) shape array.
        init_mod_file: :obj:`numpy.ndarray`
            Initial model parameters
        y_values: :obj:`numpy.ndarray`, optional
            Y positions on the CCD. If none given, defaults to the spectral
            direction pixel indices. 
        sigma: :obj:`numpy.ndarray`, optional
            Uncertainties in the y_values, for weighted fit purposes. 
        decrease_dim: int, optional
            The factor of decreased dimentionality for the fit.
            This needs to be an exact factor of the y size.

        Returns
        -------

        params: :obj:`numpy.ndarray` array
            Fitted parameters.
        """

        # FIXME More vigorous input type checking (or casting)
        if not isinstance(x_to_fit, np.ndarray):
            raise UserWarning('provided X model is not ndarray type.')
        if not isinstance(init_mod, np.ndarray):
            raise UserWarning('provided initial model is not ndarray type.')

        if x_to_fit.shape[0] % decrease_dim != 0:
            raise UserWarning(
                "Can not decrease the x value dimension by this amount. "
                "Please check if the image size in the spectral dimension "
                "is exactly divisible by this amount.")

        # Create an array of y and m values.
        x_values = x_to_fit.copy()
        order_y = np.meshgrid(np.arange(x_values.shape[1]),
                              np.arange(x_values.shape[0]) + self.m_min)
        if len(y_values) == 0:
            y_values = order_y[0]
        orders = order_y[1]

        # Allow a dimensional decrease, for speed
        if decrease_dim > 1:
            orders = np.average(orders.reshape(x_values.shape[0],
                                               x_values.shape[1] //
                                               decrease_dim, decrease_dim),
                                axis=2)
            y_values = np.average(y_values.reshape(x_values.shape[0],
                                                   # pylint: disable=maybe-no-member
                                                   x_values.shape[1] //
                                                   decrease_dim, decrease_dim),
                                  axis=2)
            x_values = np.average(x_values.reshape(x_values.shape[0],
                                                   x_values.shape[1] //
                                                   decrease_dim, decrease_dim),
                                  axis=2)

        # Flatten arrays
        orders = orders.flatten()
        y_values = y_values.flatten()  # pylint: disable=maybe-no-member
        x_values = x_values.flatten()  # pylint: disable=maybe-no-member
        sigma = sigma.flatten()

        ydeg = init_mod.shape[0] - 1
        xdeg = init_mod.shape[1] - 1
        # Do the fit!
        print("Fitting (this can sometimes take a while...)")
        from datetime import datetime
        start = datetime.now()
        init_resid = self.fit_resid(init_mod, orders, y_values, x_values,
                                    ydeg=ydeg, xdeg=xdeg, sigma=sigma)
        init_resid = (init_resid * sigma)[sigma < 1e5]
        for niter in range(maxiter):
            bestp = op.leastsq(self.fit_resid, init_mod,
                               args=(orders, y_values, x_values, ydeg, xdeg, sigma))
            final_resid = self.fit_resid(bestp[0], orders, y_values, x_values,
                                         ydeg=ydeg, xdeg=xdeg, sigma=sigma)
            rms = np.std((final_resid * sigma)[sigma < 1e5])
            bad = np.where(np.logical_and(abs(final_resid * sigma) > rms * sigma_rej,
                                          sigma < 1e5))[0]
            print(f"Time after {niter+1} iteration(s):", datetime.now()-start)
            if bad.size:
                sigma[bad] = 1e5
                init_mod = bestp[0]
                #print("REJECTING ", bad.size)
            else:
                break

        final_resid = (final_resid * sigma)[sigma < 1e5]
        params = bestp[0].reshape((ydeg + 1, xdeg + 1))
        #x_fitted = x_values - final_resid * sigma
        #for i in range(x_values.size):
        #    if sigma[i] < 1e5:
        #        # 3080 valid for red only
        #        print(f"{y_values[i]:6.1f} {x_values[i]+3080:10.5f} {x_fitted[i]+3080:10.5f} {final_resid[i]*sigma[i]:10.5f}")
        print("RMS", init_resid.std(), '->', final_resid.std())
        print("MAX", np.max(abs(init_resid)), '->', np.max(abs(final_resid)))

        # FIXME: Issues with the high resolution fit here. Why? How to diagnose?
        #import pdb; pdb.set_trace()

        return params

    def spectral_format_with_matrix(self, xmod, wavemod, spatmod=None,
                                    specmod=None, rotmod=None,
                                    return_arrays=False):
        """
        Create a spectral format, including a detector-to-slit matrix, at
        every point. 

        The input parameters represent the polynomial coefficients for second
        order descriptions of how the spectral and spatial scales vary as a
        function of order for each mode, as well as a slit rotation indicator.

        This function is crucial to the workings of the pipeline, as it is
        designed to be executed almost every time following the initialisation
        of the class as a way to define the model variables for usage in the
        pipeline.

        It takes the model parameters and defines the ``x_map``, ``w_map``,
        ``blaze`` and ``matrices`` arrays that describe every aspect of the
        spectrograph image for extraction. The two orthogonal
        ``slit_microns_per_det_pix`` variables represent the physical size of
        the full slit in detector pixels, which is scaled by the magnification
        at all points. Each (2, 2) matrix contains 2 parameters that relate to
        the magnification, and are modified by rotation angle.

        The functional form is equivalent to all other models in the spectral
        format. For the three new parameters, the simplest interpretation of the
        model files is as follows:

        | :math:`q_{00}`:  spatial/spectral/rotation scale at the reference
          order
        | :math:`q_{01}`:  variation as a function of orders divided by the
          number
          of orders
        | ... with everything else approximately zero.

        The mathematical principle behind this method is as follows. For every
        position along an order we create a matrix :math:`A_{mat}` where we map
        the input angles to output coordinates using the
        ``slit_microns_per_det_pix_x/y`` variables:

        .. math::

           A_\\textrm{mat} = \\begin{bmatrix}
                             1/(\\textrm{slit_microns_per_det_pix_y}) & 0 \\\\
                             0 & 1/\\textrm{slit_microns_per_det_pix_x}
                             \\end{bmatrix}

        We then compute an 'extra rotation matrix' :math:`R_\\textrm{mat}` which
        maps the single slit rotation value from the model for this position
        into a rotation matrix:

        .. math::

           R_\\textrm{mat} = \\begin{bmatrix}
                             \\cos(\\theta) & \\sin(\\theta) \\\\
                             -\\sin(\\theta) & \\cos(\\theta)
                             \\end{bmatrix},

        where :math:`\\theta` is the rotation of the slit in radians.

        We then do a dot product of the two matrices and invert:

        .. math::

           M = (R_\\textrm{mat} \cdot A_\\textrm{mat})^{-1}

        This is computationally complicated, but since the matrices are square
        and :math:`A_\\textrm{mat}` is diagonal, we can do this explicitly:

        .. math::

           M = \\begin{bmatrix}
                \\cos(\\theta) / slity & \\sin(\\theta) / slitx \\\\
                -\\sin(\\theta) / slity & \\cos(\\theta) / slitx
                \\end{bmatrix} ^ {-1}

        and,

        .. math::

           M = \\begin{bmatrix}
               cos(\\theta) / slitx & -sin(\\theta) / slitx \\\\
               sin(\\theta) / slity & cos(\\theta) / slity
               \\end{bmatrix}

        Therefore, in the code we do this explicitly and obtain the result with
        simple operations only.

        Notes
        -----
        Function returns are optional; instead, this function creates/updates
        class attributes that are used in other circumstances. While this is
        perhaps not advisable from a programmatic point of view, it does
        benefit from a number of advantages, as the model evaluations are used
        extensively throughout this module.

        Parameters
        ----------

        xmod: :obj:`numpy.ndarray`
            pixel position model parameters. Used in the spectral format
            function. See documentation there for more details
        wavemod: :obj:`numpy.ndarray`
            pixel position model parameters. Used in the spectral format
            function. See documentation there for more details
        spatmod: :obj:`numpy.ndarray`, optional
            Parameters from the spatial scale second order polynomial
            describing how the slit image varies in the spatial direction
            as a function of order on the CCD
        specmod: :obj:`numpy.ndarray`, optional
            Parameters from the spectral scale second order polynomial
            describing how the slit image varies in the spectral direction
            as a function of order on the CCD
        rotmod: :obj:`numpy.ndarray`, optional
            Parameters from the extra rotation second order polynomial
            describing how the slit image rotation varies
            as a function of order on the CCD
        return_arrays: bool, optional
            By default, we just set internal object properties. Return the
            arrays themselves instead as an option.

        Raises
        ------
        ValueError
            Raised if neither ``xmod`` or ``wavemod`` are provided, or if none
            of ``spatmod``, ``specmod`` and ``rotmod`` are provided.

        Returns
        -------

        x: :obj:`numpy.ndarray` array, ``(norders, ny)``
            The x-direction pixel co-ordinate corresponding to each y-pixel
            and each order (m).
        w: :obj:`numpy.ndarray` array, ``(norders, ny)``
            The wavelength co-ordinate corresponding to each y-pixel and each
            order (m).
        blaze: :obj:`numpy.ndarray` array, ``(norders, ny)``
            The blaze function (pixel flux divided by order center flux)
            corresponding to each y-pixel and each order (m).
        matrices: :obj:`numpy.ndarray` array, ``(norders, ny, 2, 2)``
            2x2 slit rotation matrices, mapping output co-ordinates back
            to the slit.
        """
        if (xmod is None) and (wavemod is None):
            raise ValueError('Must provide at least one of xparams or wparams')

        if (spatmod is None) and (specmod is None) and (rotmod is None):
            raise ValueError('Must provide at least one of spatmod, specmod or '
                             'rotmod, otherwise there is no point in running '
                             'this function.')

        # Get the basic spectral format
        xbase, waves, blaze = self.spectral_format(xparams=xmod,
                                                   wparams=wavemod)
        matrices = np.zeros((xbase.shape[0], xbase.shape[1], 2, 2))
        # Initialize key variables in case models are not supplied.
        slit_microns_per_det_pix_x = np.ones(self.szy)
        slit_microns_per_det_pix_y = np.ones(self.szy)
        rotation = np.zeros(self.szy)
        yvalue = np.arange(self.szy)

        if spatmod is not None:
            slit_microns_per_det_pix_x = self.evaluate_poly(spatmod)
        if specmod is not None:
            slit_microns_per_det_pix_y = self.evaluate_poly(specmod)
        if rotmod is not None:
            rotation = self.evaluate_poly(rotmod)
        # Loop through orders
        r_rad = np.radians(rotation)
        extra_rot_mat = np.zeros((xbase.shape[0], xbase.shape[1], 2, 2))
        # This needs to be done separately because of the
        # matrix structure
        extra_rot_mat[:, :, 0, 0] = np.cos(r_rad) * \
                                    slit_microns_per_det_pix_x
        extra_rot_mat[:, :, 0, 1] = -np.sin(r_rad) * \
                                    slit_microns_per_det_pix_x
        extra_rot_mat[:, :, 1, 0] = np.sin(r_rad) * \
                                    slit_microns_per_det_pix_y
        extra_rot_mat[:, :, 1, 1] = np.cos(r_rad) * \
                                    slit_microns_per_det_pix_y
        matrices = extra_rot_mat

        self.x_map = xbase
        self.w_map = waves
        self.blaze = blaze
        self.matrices = matrices

        if return_arrays:
            return xbase, waves, blaze, matrices

    def slit_flat_convolve(self, flat, slit_profile=None):
        """
        Dummy function, returns the input flat.

        Dummy function that would take a flat field image and a slit profile
        and convolves the two in 2D. Returns result of convolution, which
        should be used for tramline fitting.
        """
        return flat

    def manual_model_adjust(self, data, xparams, model='position', wparams=None,
                            thar_spectrum=None, percentage_variation=10,
                            vary_wrt_max=True, title=None):
        """
        Interactive manual adjustment for a :any:`polyspect` module.

        Function that uses matplotlib slider widgets to adjust a polynomial
        model overlaid on top of a flat field image. In practice this will be
        overlaid on top of the result of convolving the flat with a slit
        profile in 2D which just reveals the location of the middle of the
        orders.
        
        It uses :any:`matplotlib` slider widgets to adjust a polynomial model
        representation overlaid on top of an image. A ``data`` array is provided
        containing an image, and the model parameters needed to determine the
        desired model; ``xparams`` are needed for both position and wavelength,
        and ``wparams`` are needed for the wavelength model.

        A ``model`` variable is defined and needed to distinguish between a
        representation of the order centre position or wavelength model. A
        ``thar_spectrum`` array must be provided for the wavelength model. The
        ``percentage_variation`` variable refers to the percentage of the
        parameter values that should be allowed to range in the sliders.

        This function then goes on to show a window that contains the current
        supplied model overlayed on top of the image provided, and a second
        window containing a series of sliders which the user can use to vary the
        model parameters in real time.

        More documentation for this function may be added, but in principle the
        plotting mechanism is commented and designed to be generic in terms of
        the number of parameters. The actual workings of the plotting code is
        documented in the :any:`matplotlib` documentation.

        The function returns the changed model parameters.
        
        Parameters
        ----------
        data: :obj:`numpy.ndarray`
            an array containing data to be used as a visual comparison of the
            model
        model: string, optional
            What model would you like to adjust? Either 'position' for the x
            model or 'wavelength' for the wavelength scale. Default is
            'position'
        wparams: :obj:`numpy.ndarray`, optional
            2D array containing the initial wavelength
            model parameters.
        xparams: :obj:`numpy.ndarray`
            2D array containing the initial order location model parameters.
        thar_spectrum: :obj:`numpy.ndarray`, optional
            2D array containing the thar spectrum (from the simulator code) as a
            function of wavelength.
        percentage_variation: int, optional
            How much should the percentage adjustment in each bin as a function
            of the parameter.
        vary_wrt_max: bool, optional
            Vary all parameters intelligently with a scaling of the maximum
            variation, rather than just a percentage of each.
        title: str, optional
            Figure title for visualisation purposes. Optional and defaults to
            None

        Returns
        -------
        xparams: :obj:`numpy.ndarray`
             New adjusted x parameters

        """

        # Must provide xparams
        if (xparams is None):
            raise ValueError('Must provide at least an initial xparams')

        # Grab the model to be plotted
        x_int, wave_int, blaze_int = self.spectral_format(wparams=wparams,
                                                          xparams=xparams)
        
        # define what is to be plotted
        def plot_data(model, xparams, wparams, nxbase, ygrid,
                      thar_spectrum=None):
            """
            Function used for working out and defining
            the data to be plotted as a function of purpose 

            Parameters
            ----------

            model: string
                What model is being adjusted. This is the input to the main 
                function
            xparams: :obj:`numpy.ndarray` array
                The (adjusted) position model parameters
            wparams: :obj:`numpy.ndarray` array
                The (adjusted) wavelength model parameters 
            nxbase: :obj:`numpy.ndarray` array
                The amount to add to the xbase after the spectral format
            ygrid: :obj:`numpy.ndarray` array
                The grid of y values to plot against. This needs to be modified
            in order to ensure it is the quickest plot method possible.

            Returns
            -------
            
            plot_vals: :obj:`numpy.ndarray` array
                 The values to be plotted
            """
            xbase, wave, blaze = self.spectral_format(wparams=wparams,
                                                      xparams=xparams)
            if model == 'position':
                return ygrid.flatten()[::10], xbase.flatten()[::10] + (
                nxbase // 2)
            elif model == 'wavelength':
                # This ensures that the thorium argon interpolation is within
                # range
                thar_spectrum[0][0] = wave.min()
                thar_spectrum[0][-1] = wave.max()
                thar_spectrum[1][0] = 0.0
                thar_spectrum[1][-1] = 0.0
                interp_thar = interp1d(thar_spectrum[0], thar_spectrum[1])
                flux = interp_thar(wave)
                # Now remove anything that is below 40 times the average sigma.
                # This means only the very brightest lines are displayed.
                thar_threshold = np.average(flux) * 40.
                ygrid_filtered = ygrid[np.where(flux > thar_threshold)]
                xbase_filtered = xbase[np.where(flux > thar_threshold)]
                wave_filtered = wave[np.where(flux > thar_threshold)]
                #import pdb; pdb.set_trace()
                return ygrid_filtered.flatten(), xbase_filtered.flatten() + (
                    nxbase // 2), wave_filtered.flatten()
            else:
                raise UserWarning('invalid model type for plot_data')

        nxbase = data.shape[0]

        # Start by setting up the graphical part for the image
        fig, axx = plt.subplots()
        plt.subplots_adjust(left=0.15, bottom=0.25)
        if title is not None:
            axx.set_title(title)
        axcolor = 'lightgoldenrodyellow'

        ygrid = np.meshgrid(np.arange(data.shape[1]),
                            np.arange(x_int.shape[0]))[0]
        # The model data must be flattened for the sliders to work.
        # Then plot it!
        to_plot = plot_data(model, xparams, wparams, nxbase, ygrid,
                            thar_spectrum)
        lplot, = plt.plot(to_plot[0], to_plot[1],
                          color='green', linestyle='None', marker='.')

        #*** Plot additional colours. Can't figure out how to make this update on
        #    slider change though ***
        #if len(to_plot)>2:
        #    plt.scatter(to_plot[0], to_plot[1], c=to_plot[2])
        #    plt.colorbar() 
        
        # Now over plot the image and add a contrast adjustment slider.
        image_min = np.percentile(data,10) #!!! MJI Hack
        image_max = data.max()
        image_diff = image_max - image_min
        init_contrast = 0.8
        contrastSlider_ax  = fig.add_axes([0.15, 0.1, 0.7, 0.05])
        contrastSlider = Slider(contrastSlider_ax, 'contrast', 0, 1,
                                valinit=init_contrast)

        image = axx.imshow(data,
                           vmin = image_min, # + init_contrast*image_diff//8
                           vmax = image_max - init_contrast*image_diff, cmap='binary') #//2


        def update_imshow(val):
            """
            Function used to trigger update on the contrast slider
            """
            image.set_clim(vmin = image_min, # + contrastSlider.val*image_diff//8
                           vmax = image_max - contrastSlider.val*image_diff) #//2

        contrastSlider.on_changed(update_imshow)

        # Create a second window for sliders.
        slide_fig = plt.figure()

        # This function is executed on each slider change.
        # spectral_format is updated.
        def update(val):
            """ Function used to trigger updates on sliders """
            for i in range(npolys):
                for j in range(polyorder):
                    params[i, j] = sliders[i][j].val
            if model == 'position':
                to_plot = plot_data(model, params, wparams, nxbase, ygrid,
                                    thar_spectrum)
            elif model == 'wavelength':
                to_plot = plot_data(model, xparams, params, nxbase, ygrid,
                                    thar_spectrum)
            lplot.set_xdata(to_plot[0])
            lplot.set_ydata(to_plot[1])
            fig.canvas.draw_idle()

        if model == 'position':
            params = xparams
        elif model == 'wavelength':
            params = wparams

        # The locations of the sliders on the window depend on the number of
        # parameters
        polyorder = params.shape[1]
        npolys = params.shape[0]
        # Now we start putting sliders in depending on number of parameters
        height = 1. / (npolys * 2)
        width = 1. / (polyorder * 2)
        # Use this to adjust in a percentage how much to let each parameter
        # vary
        frac_params = np.absolute(params * (percentage_variation / 100))
        if vary_wrt_max:
            for i in range(npolys):
                frac_params[i] = np.max(
                    frac_params[-1]) / (nxbase / 2.0) ** (npolys - 1 - i)
        axq = [[0 for x in range(polyorder)] for y in range(npolys)]
        sliders = [[0 for x in range(polyorder)] for y in range(npolys)]
        # Now put all the sliders in the new figure based on position in the
        # array.
        # This double loop looks complicated, but is really a method to
        # place the right sliders in the right positions on the window
        for i in range(npolys):
            for j in range(polyorder):
                left = j * width * 2
                bottom = 1 - (i + 1) * height * 2 + height
                axq[i][j] = plt.axes([left, bottom, width, height],
                                     facecolor=axcolor) #axisbg
                if params[i, j] == 0:
                    sliders[i][j] = Slider(axq[i][j],
                                           'coeff' + str(i) + str(j), 0, 0.1,
                                           valinit=params[i, j])
                else:
                    sliders[i][j] = Slider(axq[i][j], 'coeff' + str(i) + str(j),
                                           params[i, j] - frac_params[i, j],
                                           params[i, j] + frac_params[i, j],
                                           valinit=params[i, j])
                plt.legend(loc=3)
                sliders[i][j].on_changed(update)

        # Have a button to output the current model to the correct file
        submitax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(submitax, 'Submit', color=axcolor, hovercolor='0.975')

        # This is the triggered function on submit.
        # Currently only works for the xmod but should be generalised
        def submit(event):
            """Function for the button tasks"""
            plt.close('all')
            return params

        button.on_clicked(submit)

        plt.show()
        print(params)
        return params


class WaveModel(Fittable2DModel):
    """An astropy.modeling.Model subclass to hold the WAVEMOD"""
    _param_names = ()

    def __init__(self, name=None, meta=None, model=None, arm=None):
        """
        Set up a WAVEMOD for later fitting. It must be instantiated with an
        initial set of parameters (they could be all zeros) so that the
        degrees of the polynomials are known.

        Parameters
        ----------
        name: str/None
            name of model
        meta: dict-like/None
            additional items for the model
        model: array
            starting parameters for the WAVEMOD
        arm: GhostArm instance
            for the arm being fit
        """
        self.y_degree, self.x_degree = model.shape
        self.m_ref = arm.m_ref
        self.szy = arm.szy
        self._param_names = self._generate_coeff_names()

        for param_name in self._param_names:
            self._parameters_[param_name] = Parameter(param_name,
                                                      default=np.array(0.))
        super().__init__(name=name, meta=meta,
                         **dict(zip(self._param_names, model.flatten())))

    @property
    def param_names(self):
        return self._param_names

    def _generate_coeff_names(self):
        names = []
        for j in range(self.y_degree):
            for i in range(self.x_degree):
                names.append(f'c{i}_{j}')
        return tuple(names)

    def evaluate(self, x, y, *parameters):
        # x is pixel positions, y is orders
        y_values, orders = x, y
        params = np.array(parameters).reshape(self.y_degree, self.x_degree)
        mprime = float(self.m_ref) / orders - 1
        polynomials = np.empty((len(orders), self.y_degree))
        # Find the polynomial coefficients for each order.
        for i in range(self.y_degree):
            polyq = np.poly1d(params[i, :])
            polynomials[:, i] = polyq(mprime)
        evaluation = np.empty(y_values.shape)
        # The evaluate as a function of position.
        for i in range(len(orders)):
            polyp = np.poly1d(polynomials[i, :])
            evaluation[i] = polyp(y_values[i] - self.szy // 2)
        return evaluation

    def plotit(self, x, y, waves, mask=None, filename='wavecal.pdf'):
        colors = 'rgbcmk'
        if mask is None:
            mask = np.zeros_like(x, dtype=bool)
        # waves, orders, final_resid
        fig, ax = plt.subplots()
        orders = np.unique(y).astype(int)
        residuals = self.evaluate(x, y, *self.parameters) - waves
        rms = np.std(residuals[~mask])
        for i, order in enumerate(orders):
            col = colors[i % len(colors)]
            indices = np.where(y == order)[0]
            px = x[indices]
            py = residuals[indices]
            pm = mask[indices]
            #print(order, np.median(lines[indices,2]), [(pix, resid) for pix, resid in zip(x, y)])
            ax.plot([0, self.szy], [order, order], f'{col}-', linewidth=1)
            for pix, resid, m in zip(px, py, pm):
                linestyle = "dashed" if m else "solid"
                ax.plot([pix, pix], [order, order + resid / (3 * rms)], c=col,
                        ls=linestyle, linewidth=2)
            #ax.plot(x, [order] * x.size + y / 0.002, 'bo')
        ax.set_xlim(0, self.szy)
        if self.m_ref < 60:
            ax.set_ylim(max(orders) + 2, min(orders) - 2)
        else:
            ax.set_ylim(min(orders) - 2, max(orders) + 2)
        if self.name is not None:
            ax.set_title(self.name)
        ax.set_xlabel("Column number")
        ax.set_ylabel("Order number")
        fig.savefig(filename, bbox_inches='tight')