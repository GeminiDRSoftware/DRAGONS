# Copyright(c) 2016 Association of Universities for Research in Astronomy, Inc.
# by James E.H. Turner.

import numpy as np
from astropy.modeling import Model, models, fitting
from astropy.stats import sigma_clip

from .astromodels import UnivariateSplineWithOutlierRemoval
from .astrotools import cartesian_regions_to_slices

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


def fit_1D(image, weights=None, function='legendre', order=1, axis=-1,
           lsigma=3.0, hsigma=3.0, iterations=0, grow=False, regions=None,
           plot=False):
    """
    A routine for evaluating the result of fitting 1D polynomials to each
    vector along some axis of an N-dimensional image array, with iterative
    pixel rejection and re-fitting, similar to IRAF's fit1d.

    Interactivity is currently not supported, other than plotting for
    debugging purposes. This still needs to be made to work for a single 1D
    input array.

    Parameters
    ----------

    image : array-like
        N-dimensional input array containing the values to be fitted. If
        it is a `numpy.ma.MaskedArray`, any masked points are ignored when
        fitting.

    weights : `ndarray`, optional
        N-dimensional input array containing fitting weights for each point.
        The weights will be ignored for a given 1D fit if all zero, to allow
        processing regions without data (in progress: splines only).

    function : {'legendre', 'chebyshev', 'polynomial', 'splineN'}, optional
        Fitting function/model type to be used (current default 'legendre').
        The spline options may be 'spline1' (linear) to 'spline5' (quintic).

    order : `int` or `None`, optional
        Order (number of terms or degree+1) of the fitting function (default
        1). For spline fits, this is the maximum number of spline pieces,
        which (if applicable) will be reduced in proportion to the number of
        masked pixels for each fit. A value of `None` uses as many pieces as
        required to get chi^2=1 for spline fits, while for other functions it
        maps to the default of 1.

    axis : `int`, optional
        Array axis along which to perform fitting (Python convention;
        default -1, ie. rows).

    lsigma, hsigma : `float`, optional
        Rejection threshold in standard deviations below and above the mean,
        respectively (default 3.0).

    iterations : `int`, optional
        Maximum number of rejection and re-fitting iterations (default 0, ie.
        a single fit with no iteration).

    grow : float or `False`, optional
        Distance within which to extend rejection to the neighbours of
        statistically-rejected pixels (eg. 1 for the nearest neighbours).

    regions : `str`, optional
        One or more comma-separated pixel ranges to be used in fitting each
        1D model. Each range is 1-indexed, inclusive of the upper limit and
        may use a colon or hyphen separator. An upper or lower limit may be
        omitted, to use the remainder of the axis. The default of `None` (or
        '*' or '') causes the entire axis to be used. Any data outside the
        specified range(s) are ignored in fitting (even if those ranges are
        completely masked in the input `image`).

    plot : bool
        Plot the images if True, default is False.

    Returns
    -------

    `ndarray`
        An array of the same shape as the input, whose values are evaluated
        from the polynomial fits to each 1D vector.

    """

    # Convert array-like input to a MaskedArray internally, to ensure it's an
    # `ndarray` instance and that any mask gets passed through to the fitter:
    image = np.ma.masked_array(image)

    # Determine how many pixels we're fitting each vector over:
    try:
        npix = image.shape[axis]
    except IndexError:
        raise ValueError('axis={0} out of range for input shape {1}'
                         .format(axis, image.shape))

    # Define pixel grid to fit on:
    points = np.arange(npix, dtype=np.int16)

    # Define the model to be fitted:
    func, funcargs = function_map[function]
    try:
        astropy_model = issubclass(func, Model)
    except TypeError:
        astropy_model = False

    # The spline fitting uses an adaptive criterion when order=None, but the
    # AstroPy polynomials require an integer degree so map None to default:
    if order is None and func is not UnivariateSplineWithOutlierRemoval:
        order = 1

    # Parse the sample regions:
    slices = cartesian_regions_to_slices(regions)

    # Convert user regions to a Boolean mask for slicing:
    user_reg = np.zeros(npix, dtype=np.bool)
    for _slice in slices:
        user_reg[_slice] = True

    # To support fitting any axis of an N-dimensional array, we must flatten
    # all the other dimensions into a single "model set axis" first; it's
    # about 10-15% more efficient to stack astropy models along the second
    # axis (for a 3k x 2k image), because that's what the linear fitter does
    # internally, but for general callable functions we stack along the first
    # axis in order to loop over the fits easily:
    if astropy_model:
        ax_before = 0
        stack_shape = (npix,) if image.size == npix else (npix, -1)
    else:
        ax_before = image.ndim
        stack_shape = (-1, npix)
    image = np.rollaxis(image, axis, ax_before)
    tmpshape = image.shape
    image = image.reshape(stack_shape)
    if weights is not None:
        weights = np.rollaxis(weights, axis, ax_before).reshape(stack_shape)

    # Create an empty, full-sized mask within which the fitter will populate
    # only the user-specified region(s):
    mask = np.zeros(image.shape, dtype=bool)

    # Initialize an array with same dtype as input to accumulate fit values
    # into when looping & because modelling always seems to return float64:
    fitvals = np.zeros_like(image.data)

    # Branch pending on whether we're using an AstroPy model or some other
    # supported fitting function (principally splines):
    if astropy_model:

        # A single model is treated specially because FittingWithOutlierRemoval
        # fails for "1-model sets" and it should be more efficient anyway:
        image_to_fit = image
        if image.ndim == 1:
            n_models = 1
        elif image.mask is np.ma.nomask:
            n_models = image.shape[1]
        else:
            # remove fully masked columns otherwise this will lead to
            # Runtime warnings from Numpy because of divisions by zero.
            good_cols = (~image.mask).sum(axis=0) > order
            n_models = np.sum(good_cols)
            if n_models < image.shape[1]:
                image_to_fit = image[:, good_cols]
                weights = weights[:, good_cols]

        model_set = func(degree=(order - 1), n_models=n_models,
                         model_set_axis=(None if n_models == 1 else 1),
                         **funcargs)

        # Configure iterative linear fitter with rejection:
        fitter = fitting.FittingWithOutlierRemoval(
            fitting.LinearLSQFitter(),
            sigma_clip,
            niter=iterations,
            # additional args are passed to the outlier_func, i.e. sigma_clip
            sigma_lower=lsigma,
            sigma_upper=hsigma,
            maxiters=1,
            cenfunc='mean',
            stdfunc='std',
            grow=grow  # requires AstroPy 4.2 (#10613)
        )

        # Fit the pixel data with rejection of outlying points:
        fitted_model, fitted_mask = fitter(
            model_set,
            points[user_reg], image_to_fit[user_reg],
            weights=None if weights is None else weights[user_reg]
        )

        # Determine the evaluated model values we want to return:
        if image.ndim > 1 and n_models < image.shape[1]:
            # If we removed bad columns, we now need to fill them properly
            # in the output array
            fitvals[:, good_cols] = fitted_model(points, model_set_axis=False)
            # this is quite ugly, but seems the best way to assign to an
            # array with a mask on both dimensions. This is equivalent to:
            #   mask[user_reg, masked_cols] = fitted_mask
            mask[user_reg[:, None] & good_cols] = fitted_mask.flat
        else:
            fitvals[:] = fitted_model(points, model_set_axis=False)
            mask[user_reg] = fitted_mask

    else:

        # If there are no weights, produce a None for every row:
        weights = iter(lambda: None, True) if weights is None else weights

        user_masked = ~user_reg

        for n, (imrow, wrow) in enumerate(zip(image, weights)):

            # Deal with weights being None or all undefined in regions without
            # data (should we allow for NaNs as well?):
            wrow = wrow[user_reg] if wrow is not None and np.any(wrow) else None

            # Could generalize this a bit further using another dict to map
            # parameter names in function_map, eg. weights -> w?
            fitted_model = func(points[user_reg], imrow[user_reg], order=order,
                                w=wrow, niter=iterations, grow=int(grow),
                                lsigma=lsigma, hsigma=hsigma, maxiters=1,
                                **funcargs)

            # Determine model values to be returned. This is somewhat specific
            # to the spline fitter and using fitted_model.data only improves
            # performance by ~1% compared with just re-evaluating everything,
            # but it's the only way to get the mask:
            fitvals[n][user_reg] = fitted_model.data
            mask[n][user_reg] = fitted_model.mask
            fitvals[n][user_masked] = fitted_model(points[user_masked])
            # fitvals[n] = fitted_model(points)

    # TEST: Plot the fit:
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        points1 = points+1
        if astropy_model:
            idx = slice(None) if n_models == 1 else \
                  (slice(None), image.shape[1] // 4)
        else:
            idx = (image.shape[0] // 4, slice(None))
        imrow, maskrow, fitrow = image.data[idx], mask[idx], fitvals[idx]
        ax.plot(points1, imrow, 'b.')
        ax.plot(points1[maskrow], imrow[maskrow], 'r.')
        ax.plot(points1, fitrow, 'k')

        # Find starting index of each selected or omitted range in user_reg
        # (considering regions outside the array False and starting from the
        # first True), then subtract 1 from every second index, pair them to
        # get (start, end) ranges and convert back to 1-indexing for display.
        # Don't use slice objects directly because their limits can be None,
        # adjacent or overlapping, whereas here we want a minimal set of
        # unique ranges as applied to the data:
        reg_changes = np.where(np.concatenate((user_reg[:1],
                                               user_reg[1:] != user_reg[:-1],
                                               user_reg[-1:])))[0]
        mixed_coords = ax.get_xaxis_transform()  # x in data units, y in frac.
        # The extra +0.001 here avoids having identical start & end values for
        # single-pixel ranges, otherwise they don't get drawn:
        for start, end in zip(reg_changes[::2]+1, reg_changes[1::2]+0.001):
            ax.annotate('', xy=(start, 0), xytext=(end, 0),
                        xycoords=mixed_coords, textcoords=mixed_coords,
                        arrowprops=dict(arrowstyle='|-|,widthA=0.5,widthB=0.5',
                                        shrinkA=0., shrinkB=0., color='green'))

        plt.show()

    # Restore the ordering & shape of the input array:
    fitvals = fitvals.reshape(tmpshape)
    if astropy_model:
        fitvals = np.rollaxis(fitvals, 0, (axis + 1) or fitvals.ndim)
    else:
        fitvals = np.rollaxis(fitvals, -1, axis)

    return fitvals
