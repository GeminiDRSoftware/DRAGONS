# Copyright(c) 2016,2018,2020-2022 Association of Universities for Research in Astronomy, Inc.

from collections.abc import Iterable

import numpy as np
from astropy.modeling import Model, models, fitting
from astropy.stats import sigma_clip

from .astromodels import UnivariateSplineWithOutlierRemoval
from . import astrotools as at

__all__ = ['fit_1D']

function_map = {
    'chebyshev': (models.Chebyshev1D, {}),
    'legendre': (models.Legendre1D, {}),
    'polynomial': (models.Polynomial1D, {}),
    'spline1': (UnivariateSplineWithOutlierRemoval, {'k': 1}),
    'spline2': (UnivariateSplineWithOutlierRemoval, {'k': 2}),
    'spline3': (UnivariateSplineWithOutlierRemoval, {'k': 3}),
    'spline4': (UnivariateSplineWithOutlierRemoval, {'k': 4}),
    'spline5': (UnivariateSplineWithOutlierRemoval, {'k': 5}),
}


class fit_1D:
    """
    A class for evaluating the result of fitting 1D polynomials to each
    vector along some axis of an N-dimensional image array, with iterative
    pixel rejection and re-fitting, similar to IRAF's fit1d.

    Only simple, non-interactive plotting for inspection/debugging is
    implemented here; DRAGONS provides interactive plotting via a separate
    wrapper.


    Parameters
    ----------

    image : array-like
        N-dimensional input array containing the values to be fitted. If
        it is a `numpy.ma.MaskedArray`, any masked points are ignored when
        fitting.

    points : `~numpy.ndarray`, optional
        1-dimensional input array with the x-values of each 1D slice being
        fitted, If not given, the independent variable will be treated as
        a sequence of integers starting at zero.

    weights : `~numpy.ndarray`, optional
        N-dimensional input array containing fitting weights for each point.
        The weights will be ignored for a given 1D fit if all zero, to allow
        processing regions without data (in progress: splines only). If the
        data have known uncertainties, the weights should be equal to the
        inverse of the standard deviation.

    function : {'legendre', 'chebyshev', 'polynomial', 'splineN'}, optional
        Fitting function/model type to be used (current default 'legendre').
        The spline options may be 'spline1' (linear) to 'spline5' (quintic).

    domain : tuple, optional
        Domain for any of the astropy polynomial models. If None, the model
        is fit with a domain equal to the interval covered by the independent
        variable.

    order : `int` or `None`, optional
        Order/degree (ie. highest power, or number of terms minus one) for
        polynomial fitting functions. For spline fits, this is the maximum
        number of spline pieces, which (if applicable) will be reduced in
        proportion to the number of masked pixels for each fit. The default of
        0 always fits a constant. A value of `None` uses as many pieces as
        required to get chi^2=1 for spline fits, while for other functions it
        maps to the default of 0. Note that the equivalent value for polynomial
        functions in IRAF is 1 greater than with this definition.

    axis : `int`, optional
        Array axis along which to perform fitting (Python convention;
        default -1, ie. rows).

    sigma_lower, sigma_upper : `float`, optional
        Rejection threshold in standard deviations below and above the mean,
        respectively (default 3.0).

    niter : `int`, optional
        Maximum number of rejection and re-fitting iterations (default 0, ie.
        a single fit with no iteration).

    grow : `float` or `False`, optional
        Distance within which to extend rejection to the neighbours of
        statistically-rejected pixels (eg. 1 for the nearest neighbours).

    regions : `str` or list of `slice`, optional
        One or more comma-separated pixel ranges to be used in fitting each
        1D model. If this is a string, each range is 1-indexed, inclusive of
        the upper limit and may use a colon or hyphen separator. An upper or
        lower limit may be omitted, to use the remainder of the axis. The
        default of `None` (or '*' or '') causes the entire axis to be used.
        Any data outside the specified range(s) are ignored in fitting (even
        if those ranges are completely masked in the input `image`). If the
        argument is a list of slice objects, they follow the usual Python
        indexing conventions.

    plot : `bool` or `tuple` of `int`
        Plot a sample 1D fit to `image`? If `True`, the central (rounded up)
        1D fit will be plotted. A different fit can be selected by providing a
        co-ordinate tuple (using Python convention) with the fitted `axis`
        omitted, eg. (24, 34) would fit the central spectrum of an IFU data cube
        with shape (2822, 49, 68). For 2D data only, a single row/column number
        may be provided without parentheses. The default is `False` (no plot).

    Attributes
    ----------

    The arguments that control fitting and rejection (`function`, `order`,
    `axis`, `sigma_lower`, `sigma_upper`, `niter`, `grow` and `regions`) are
    stored in like-named attributes, in addition to which the following are
    defined:

    mask : `~numpy.ndarray`
        A Boolean mask matching the input `image` that records which pixels
        (`True`) were ignored when fitting, either because they were already
        masked in the input `image` or because they were rejected statistically
        during the iterative fitting process.

    model_class : `~astropy.modeling.core.Model` or
                  `astromodels.UnivariateSplineWithOutlierRemoval:`
        The underlying class to which the `function` argument has been mapped.

    regions_pix : `tuple` of (`int`, `int`)
        The actual fitted pixel ranges of the input `image` corresponding to
        `regions`, in inclusive, 1-indexed (start, end) pairs, with no
        wildcards or adjacent/overlapping ranges. These are provided primarily
        to facilitate plot annotation.

    """

    def __init__(self, image, points=None, weights=None, function='chebyshev',
                 domain=None,
                 order=0, axis=-1, sigma_lower=3.0, sigma_upper=3.0, niter=0,
                 grow=False, regions=None, plot=False):

        # Save the fitting parameter values:
        self.points = None if points is None else np.asarray(points)
        self.function = function
        self.domain = domain
        self.order = order
        self.axis = axis
        self.sigma_lower = sigma_lower
        self.sigma_upper = sigma_upper
        self.niter = niter
        self.grow = grow
        self.regions = regions

        # Map the specified fitting function to a model:
        self.model_class, self.model_args = function_map[self.function]

        # The spline fitting uses an adaptive criterion when order=None, but
        # AstroPy polynomials require an integer degree so map None to default:
        if (self.order is None and
            self.model_class is not UnivariateSplineWithOutlierRemoval):
            self.order = 0

        # Parse the sample regions or check that they're already slices:
        if not regions or isinstance(regions, str):
            self._slices = (at.cartesian_regions_to_slices(regions)
                            if points is None else
                            at.parse_user_regions(regions, dtype=self.points.dtype))
        else:
            self._slices = None
            try:
                if all(isinstance(item, slice) for item in regions):
                    self._slices = regions
            except TypeError:
                pass
            if self._slices is None:
                raise TypeError('regions must be a string or a list of slices')

        # Perform & save the fits:
        self._fit(image, weights=weights, plot=plot)

    def _fit(self, image, weights=None, plot=False):

        # Convert array-like input to a MaskedArray internally, to ensure it's
        # an `ndarray` instance and that any mask gets passed through to the
        # fitter:
        origim = image  # plotting works with the original image co-ords
        image = np.ma.masked_array(image)

        # Determine how many pixels we're fitting each vector over:
        try:
            npix = image.shape[self.axis]
        except IndexError:
            raise ValueError('axis={0} out of range for input shape {1}'
                             .format(self.axis, image.shape))

        # Define pixel grid to fit on:
        points = (np.arange(npix, dtype=np.int16) if self.points is None
                  else self.points)
        if self.domain is None:
            self.domain = (min(points), max(points))

        # Classify the model to be fitted:
        try:
            astropy_model = issubclass(self.model_class, Model)
        except TypeError:
            astropy_model = False

        # Convert user regions to a Boolean mask for slicing:
        user_reg = ~at.create_mask_from_regions(points, self._slices)

        # Record the actual fitted, inclusive ranges in 1-indexed pixels for
        # this dataset (with no wildcards or adjacent/overlapping ranges etc.),
        # mostly for plot annotation. Find the starting index of each selected
        # or omitted range in user_reg (considering regions outside the array
        # False and starting from the first True), then subtract 1 from every
        # second index, pair them to get (start, end) ranges and add back the
        # origin of 1:
        reg_changes = np.where(
            np.concatenate((user_reg[:1],
                            user_reg[1:] != user_reg[:-1],
                            user_reg[-1:]))
        )[0]
        self.regions_pix = tuple((start, end) for start, end in
                                 zip(reg_changes[::2]+1, reg_changes[1::2]))
        self.fitted_regions = tuple((points[start], points[end])
                                    for start, end in zip(reg_changes[::2],
                                                          reg_changes[1::2]-1))

        # To support fitting any axis of an N-dimensional array, we must
        # flatten all the other dimensions into a single "model set axis"
        # first; it's about 10-15% more efficient to stack astropy models along
        # the second axis (for a 3k x 2k image), because that's what the linear
        # fitter does internally, but for general callable functions we stack
        # along the first axis in order to loop over the fits easily:
        if astropy_model:
            ax_before = 0
            stack_shape = (npix,) if image.size == npix else (npix, -1)
        else:
            ax_before = image.ndim
            stack_shape = (-1, npix)
        image = np.rollaxis(image, self.axis, ax_before)
        self._tmpshape = image.shape
        image = image.reshape(stack_shape)
        if weights is not None:
            weights = np.rollaxis(np.array(weights), self.axis,
                                  ax_before).reshape(stack_shape)

        # Record intermediate array properties for later evaluation of models
        # onto the same grid, where only a subset of rows/cols may have been
        # fitted due to some being entirely masked:
        self._stack_shape = image.shape
        self._dtype = image.dtype if image.dtype.kind == 'f' else np.float32

        # Create a full-sized mask within which the fitter will populate the
        # user-specified region(s) and the rest will remain masked out:
        mask = np.ones(image.shape, dtype=bool)

        # Branch pending on whether we're using an AstroPy model or some other
        # supported fitting function (principally splines):
        if astropy_model:

            # Treat single model specially because FittingWithOutlierRemoval
            # fails for "1-model sets" and it should be more efficient anyway:
            image_to_fit = image
            if image.ndim == 1:
                if (image.mask is np.ma.nomask or
                      (~image.mask).sum() > self.order):
                    n_models = 1
                else:
                    n_models = 0  # don't try to fit a fully-masked 1D array
            elif image.mask is np.ma.nomask:
                n_models = image.shape[1]
            else:
                # remove fully masked columns otherwise this will lead to
                # Runtime warnings from Numpy because of divisions by zero.
                self._good_cols = (~image.mask).sum(axis=0) > self.order
                n_models = np.sum(self._good_cols)
                if n_models < image.shape[1]:
                    image_to_fit = image[:, self._good_cols]
                    if weights is not None:
                        weights = weights[:, self._good_cols]
                # remove model set axis if this leaves a single / no model
                if image_to_fit.shape[1] < 2:
                    image_to_fit = image_to_fit.reshape((image_to_fit.size,))
                    if weights is not None:
                        weights = weights.reshape((weights.size,))

            # Do the fitting unless the input data are entirely masked, which
            # can happen eg. when inspecting the fit for a sample image row.
            if n_models == 0:
                fitted_models = None

            else:
                model_set = self.model_class(
                    degree=self.order, n_models=n_models,
                    domain=self.domain,
                    model_set_axis=(None if n_models == 1 else 1),
                    **self.model_args
                )

                # Configure iterative linear fitter with rejection:
                fitter = fitting.FittingWithOutlierRemoval(
                    fitting.LinearLSQFitter(),
                    sigma_clip,
                    niter=self.niter,
                    # additional args are passed to outlier_func (sigma_clip)
                    sigma_lower=self.sigma_lower,
                    sigma_upper=self.sigma_upper,
                    maxiters=1,
                    cenfunc='mean',
                    stdfunc='std',
                    grow=self.grow  # requires AstroPy 4.2 (#10613)
                )

                # Fit the pixel data with rejection of outlying points:
                fitted_models, fitted_mask = fitter(
                    model_set,
                    points[user_reg], image_to_fit[user_reg],
                    weights=None if weights is None else weights[user_reg]
                )
                self.fit_info = fitter.fit_info

                # Incorporate mask for fitted columns into the full-sized mask:
                if image.ndim > 1 and n_models < image.shape[1]:
                    # this is quite ugly, but seems the best way to assign
                    # to an array with a mask on both dimensions. This is
                    # equivalent to:
                    #   mask[user_reg, masked_cols] = fitted_mask
                    mask[user_reg[:, None] & self._good_cols] = fitted_mask.flat
                else:
                    mask[user_reg] = fitted_mask

        else:
            #max_order = len(points) - self.model_args["k"]
            #if self.order is not None and self.order > max_order:
            #    self.order = max_order

            # If there are no weights, produce a None for every row:
            weights = iter(lambda: None, True) if weights is None else weights

            fitted_models = []

            for n, (imrow, wrow) in enumerate(zip(image, weights)):

                # Deal with weights being None or all undefined in regions
                # without data (should we allow for NaNs as well?). The scipy
                # spline-fitting routines state that weights should be inverse
                # standard deviation, whereas fit_1D takes inverse-variance.
                wrow = (wrow[user_reg] if wrow is not None and np.any(wrow)
                        else None)

                # Could generalize this a bit further using another dict to map
                # parameter names in function_map, eg. weights -> w?
                single_model = self.model_class(
                    points[user_reg], imrow[user_reg], order=self.order,
                    w=wrow, niter=self.niter, grow=int(self.grow),
                    sigma_lower=self.sigma_lower, sigma_upper=self.sigma_upper,
                    maxiters=1, **self.model_args
                )
                fitted_models.append(single_model)

                # Retrieve the mask from the spline object, but discard the
                # fitted values because they only apply to a subsection
                # (user_reg) of the original grid, rather than whatever points
                # the user might request, and re-using them where possible only
                # improves performance by ~1% for a single fit & evaluation:
                mask[n][user_reg] = single_model.mask
                single_model.data = None
            self.fit_info = single_model.fit_info

        # Save the set of fitted models in the flattened co-ordinate system,
        # to allow later (re-)evaluation at arbitrary points:
        self._models = fitted_models

        # Convert the mask to the ordering & shape of the input array and
        # save it. Calculate rms. Suppress warnings for no data points
        mask = mask.reshape(self._tmpshape)
        with np.errstate(invalid="ignore", divide="ignore"):
            if astropy_model:
                start = (self.axis + 1) or mask.ndim
                self.mask = np.rollaxis(mask, 0, start)
                rms = (np.rollaxis(image.reshape(self._tmpshape), 0, start) -
                       self.evaluate())[~self.mask].std()
            else:
                self.mask = np.rollaxis(mask, -1, self.axis)
                rms = (np.rollaxis(image.reshape(self._tmpshape), -1, self.axis) -
                       self.evaluate())[~self.mask].std()
        self.rms = rms if rms is not np.ma.masked else np.nan

        # Plot the fit:
        if plot is not False:
            self._plot(origim, index=None if plot is True else plot)

    # Basic plot for debugging/inspection (interactive plotting will be handled
    # by a separate wrapper in DRAGONS):
    def _plot(self, data, index=None):

        import matplotlib.pyplot as plt

        # Convert index to a tuple that can be used for slicing the arrays,
        # selecting the (approximately) central 1D model by default:
        if index is None:
            index = [axlen//2 for axlen in data.shape]
            index[self.axis] = slice(None)  # easiest way for negative indices
        else:
            axis = self.axis + data.ndim if self.axis < 0 else self.axis
            index = list(index) if isinstance(index, Iterable) else [index]
            index.insert(axis, slice(None))
            if len(index) != data.ndim:
                raise ValueError('index should have 1 dimension fewer than '
                                 'data')
        index = tuple(index)  # NumPy warns against indexing with a list

        points1 = (np.arange(data.shape[self.axis], dtype=np.int16)
                   if self.points is None else self.points)
        imrow, maskrow = np.ma.masked_array(data).data[index], self.mask[index]
        # This is a bit grossly inefficient, but currently there's no way to
        # evaluate a single model from a set and it's only a "debugging" plot
        # (if needed later, we could try constructing a single model by slicing
        # the model set parameters):
        fitrow = self.evaluate()[index]

        fig, ax = plt.subplots()

        ax.plot(points1, imrow, 'b.')
        ax.plot(points1[maskrow], imrow[maskrow], 'r.')
        ax.plot(points1, fitrow, 'k')

        mixed_coords = ax.get_xaxis_transform()  # x data units, y frac.

        # The extra +0.001 here avoids having identical start & end values
        # for single-pixel ranges, otherwise they don't get drawn:
        for start, end in self.fitted_regions:
            ax.annotate('', xy=(start, 0), xytext=(end+0.001, 0),
                        xycoords=mixed_coords, textcoords=mixed_coords,
                        arrowprops=dict(
                            arrowstyle='|-|,widthA=0.5,widthB=0.5',
                            shrinkA=0., shrinkB=0., color='green'
                        ))

        plt.show()

    def evaluate(self, points=None):
        """
        Evaluate the fitted model.

        Parameters
        ----------

        points : array-like, optional
            1D input array containing the 0-indexed points along `self.axis`
            at which the fitted models are to be evaluated. If this argument
            is not specified, the fit will be sampled at the same points as
            the original input `image` to which it was performed.

        Returns
        -------

        fitvals : `numpy.ndarray`
            Floating-point array of fitted model values, sampled at `points`
            along the fitted `axis` and matching the shape of the original
            input `image` to which the fit was performed along any other axes.

        """
        astropy_model = isinstance(self._models, Model) or self._models is None

        tmpaxis = 0 if astropy_model else -1

        # Determine how to reproduce the correct array shape, orientation and
        # sampling from flattened model output:
        # - Does this need optimizing for simple 1D input?
        if points is None:
            points = (np.arange(self._tmpshape[tmpaxis], dtype=np.int16)
                      if self.points is None else self.points)
            tmpshape = self._tmpshape
            stack_shape = self._stack_shape
        else:
            tmpshape = list(self._tmpshape)
            tmpshape[tmpaxis] = np.asarray(points).size
            stack_shape = list(self._stack_shape)
            stack_shape[tmpaxis] = tmpshape[tmpaxis]

        # Create an output array of the same shape & type as the fitted input
        # image (except for any change of sample "points" along the fitted
        # axis) that fitted values can be accumulated into and because
        # modelling always seems to return float64:
        fitvals = np.zeros(stack_shape, dtype=self._dtype)

        # Determine the model values we want to return (if there were no good
        # columns to fit models to, just keep the already-populated zeroes):
        if self._models is not None:
            if astropy_model:
                if fitvals.ndim > 1 and len(self._models) < fitvals.shape[1]:
                    # If we removed bad columns, we now need to fill them
                    # properly in the output array
                    tmpvals = self._models(points, model_set_axis=False)
                    if tmpvals.ndim == 1:  # single model for only 1 good col
                        tmpvals = tmpvals[:, np.newaxis]
                    fitvals[:, self._good_cols] = tmpvals
                    del tmpvals
                else:
                    fitvals[:] = self._models(points, model_set_axis=False)
            else:
                for n, single_model in enumerate(self._models):
                    # Determine model values to be returned (see comment in
                    # _fit about discarding values stored in the spline object)
                    fitvals[n] = single_model(points)

        # Restore the ordering & shape of the original input array:
        fitvals = fitvals.reshape(tmpshape)
        if astropy_model:
            fitvals = np.rollaxis(fitvals, 0, (self.axis + 1) or fitvals.ndim)
        else:
            fitvals = np.rollaxis(fitvals, -1, self.axis)

        return fitvals

    @property
    def model(self):
        """
        Expose the callable instance describing the model fit

        Returns
        -------
        callable : a single callable (either an ~astropy.modeling.core.Model`
                  or `astromodels.UnivariateSplineWithOutlierRemoval:`)
                  representing the model fit
        """
        # If no models were fitted because the input is entirely masked, return
        # a model of the expected type with all zeroes, for compatibility:
        if self._models is None:
            return self.model_class(degree=self.order, n_models=1,
                                    domain=self.domain, model_set_axis=None,
                                    **self.model_args)

        if len(self._models) > 1:
            raise ValueError("Can only provide model property if there is a "
                             "single model.")
        astropy_model = isinstance(self._models, Model)
        if astropy_model:
            return self._models
        else:
            return self._models[0]

    def offset_fit(self, offset):
        """
        Applies a constant shift to the model fit.

        Parameters
        ----------
        offset: float
            amount by which all fits are to be shifted
        """
        astropy_model = isinstance(self._models, Model)
        if astropy_model:
            self._models.c0 += offset
        else:
            for spline in self._models:
                t, c, k = spline.tck
                c[:-(k+1)] += offset

    def extract_params(self):
        """
        Stand-in method by ooberdorf pending something from Chris or James.

        Convert the state of this fit_1D model into a dictionary.  This can be used
        to create a fresh fit.

        Returns
        -------

        """
        retval = dict()
        terms = ["function", "order", "sigma_lower", "sigma_upper", "niter", "grow", "regions"]
        for term in terms:
            val = getattr(self, term, None)
            if val is not None:
                retval[term] = val
        return retval

    @staticmethod
    def translate_params(params):
        """
        Translates the parameter names used by the primitives into those used
        by fit_1D. Does not do any meaningful validation since that should be
        handled by the Configs.

        Parameters
        ----------
        params : dict
            An input dict of parameters (can include irrelevant ones)

        Returns
        -------
        params : dict
            A dict suitable for passing to fit1D
        """
        lsigma, hsigma = params.get("lsigma"), params.get("hsigma")
        if lsigma is None and hsigma is None:
            lsigma = hsigma = params.get("sigma")  # still None if non-existent

        # FittingWithOutlierRemoval should support niter=None but it crashes
        niter = params.get("niter")
        if niter is None:
            niter = 100

        new_params = {"niter": niter,
                      "order": params["order"],
                      "function": params["function"],
                      "sigma_lower": lsigma,
                      "sigma_upper": hsigma,
                      "regions": params.get("regions")}
        if "grow" in params:
            new_params["grow"] = params["grow"]
        return new_params
