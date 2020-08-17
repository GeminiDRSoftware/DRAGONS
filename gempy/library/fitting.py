# Copyright(c) 2016 Association of Universities for Research in Astronomy, Inc.
# by James E.H. Turner.

import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

from .astrotools import cartesian_regions_to_slices

__all__ = ['fit_1D']

function_map = {
    'chebyshev': models.Chebyshev1D,
    'legendre': models.Legendre1D,
    'polynomial': models.Polynomial1D,
}


def fit_1D(image, weights=None, function='legendre', order=1, axis=-1,
           lsigma=3.0, hsigma=3.0, iterations=0, regions=None, plot=False):
    """
    A routine for evaluating the result of fitting 1D polynomials to each
    vector along some axis of an N-dimensional image array, with iterative
    pixel rejection and re-fitting, similar to IRAF's fit1d.

    Only a subset of fit1d functionality is currently supported (not
    including interactivity, other than for debugging purposes). This still
    needs to be made to work for a single 1D input array.

    Parameters
    ----------

    image : array-like
        N-dimensional input array containing the values to be fitted. If
        it is a `numpy.ma.MaskedArray`, any masked points are ignored when
        fitting.

    weights : `ndarray`, optional
        N-dimensional input array containing fitting weights for each point.

    function : {'legendre', 'chebyshev', 'polynomial'}, optional
        Fitting function/model type to be used (current default 'legendre').

    order : `int`, optional
        Order (number of terms or degree+1) of the fitting function
        (default 1).

    axis : `int`, optional
        Array axis along which to perform fitting (Python convention;
        default -1, ie. rows).

    lsigma, hsigma : `float`, optional
        Rejection threshold in standard deviations below and above the mean,
        respectively (default 3.0).

    iterations : `int`, optional
        Number of rejection and re-fitting iterations (default 0, ie. a single
        fit with no iteration).

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

    # Parse the sample regions:
    slices = cartesian_regions_to_slices(regions)

    # Record input dtype so we can cast the evaluated fits back to it, since
    # modelling always seems to return float64:
    intype = image.dtype

    # To support fitting any axis of an N-dimensional array, we must flatten
    # all the other dimensions into a single "model set axis" first; I think
    # it's marginally more efficient in general to stack the models along the
    # second axis, because that's what the linear fitter does internally.
    image = np.rollaxis(image, axis, 0)
    tmpshape = image.shape
    image = image.reshape(npix, -1)
    if weights is not None:
        weights = np.rollaxis(weights, axis, 0).reshape(npix, -1)

    # Define pixel grid to fit on:
    points = np.arange(npix, dtype=np.int16)
    points_2D = np.tile(points, (image.shape[1], 1)).T  # pending astropy #7317

    # Convert user regions to a Boolean mask for slicing:
    user_reg = np.zeros(npix, dtype=np.bool)
    for _slice in slices:
        user_reg[_slice] = True

    # Define the model to be fitted:
    func = function_map[function]
    model_set = func(degree=order - 1, n_models=image.shape[1],
                     model_set_axis=1)

    # Configure iterative linear fitter with rejection:
    fitter = fitting.FittingWithOutlierRemoval(
        fitting.LinearLSQFitter(),
        sigma_clip,
        niter=iterations,
        # additional args are passed to the outlier_func, i.e. sigma_clip
        sigma_lower=lsigma,
        sigma_upper=hsigma,
        cenfunc='mean',
        stdfunc='std',
        maxiters=1
    )

    # Create an empty, full-sized mask within which the fitter will populate
    # only the user-specified region(s):
    mask = np.zeros_like(image, dtype=np.bool)

    # TO DO: the fitter seems to be failing with input weights?

    # Fit the pixel data with rejection of outlying points:
    fitted_model, mask[user_reg] = fitter(model_set,
                                          points[user_reg], image[user_reg],
                                          weights=None if weights is None else
                                                  weights[user_reg])

    # Determine the evaluated model values we want to return:
    fitvals = fitted_model(points_2D).astype(intype)

    # # TEST: Plot the fit:
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        row = image.shape[1] // 4
        points1, imrow, maskrow = points+1, image.data.T[row], mask.T[row]
        ax.plot(points1, imrow, 'b.')
        ax.plot(points1[maskrow], imrow[maskrow], 'r.')
        ax.plot(points1, fitvals.T[row], 'k')

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

        # A cruder way of doing the region annotation:
        # # ax.plot(points1[user_reg], np.zeros_like(imrow[user_reg]), 'g_',
        # #         markersize=2)

        plt.show()

    # Restore the ordering & shape of the input array:
    fitvals = fitvals.reshape(tmpshape)
    fitvals = np.rollaxis(fitvals, 0, axis + 1)

    return fitvals
