# Copyright(c) 2016 Association of Universities for Research in Astronomy, Inc.
# by James E.H. Turner.

import numpy as np
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

__all__ = ['fit_1D']

function_map = {
    'chebyshev': models.Chebyshev1D,
    'legendre': models.Legendre1D,
    'polynomial': models.Polynomial1D,
}


def fit_1D(image, function='legendre', order=1, axis=-1, lsigma=3.0,
           hsigma=3.0, iterations=0, plot=False):
    """
    A routine for evaluating the result of fitting 1D polynomials to each
    vector along some axis of an N-dimensional image array, with iterative
    pixel rejection and re-fitting, similar to IRAF's fit1d.

    Only a subset of fit1d functionality is currently supported (not
    including interactivity).

    Parameters
    ----------

    image : `ndarray`
        N-dimensional input array containing the values to be fitted.

    function : {'legendre'}, optional
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

    plot : bool
        Plot the images if True, default is False.

    Returns
    -------

    `ndarray`
        An array of the same shape as the input, whose values are evaluated
        from the polynomial fits to each 1D vector.

    """

    # Determine how many pixels we're fitting each vector over:
    try:
        npix = image.shape[axis]
    except IndexError:
        raise ValueError('axis={0} out of range for input shape {1}'
                         .format(axis, image.shape))

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

    # Define pixel grid to fit on:
    points = np.arange(npix, dtype=np.int16)
    points_2D = np.tile(points, (image.shape[1], 1)).T  # pending astropy #7317

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
        cenfunc=np.ma.mean,
        stdfunc=np.ma.std,
        maxiters=1
    )

    # Fit the pixel data with rejection of outlying points:
    fitted_model, mask = fitter(model_set, points, image)

    # Determine the evaluated model values we want to return:
    fitvals = fitted_model(points_2D).astype(intype)

    # # TEST: Plot the fit:
    if plot:
        import matplotlib.pyplot as plt
        row = image.shape[1] // 4
        plt.plot(points, mask.T[row], 'm.')
        plt.plot(points, image.T[row], 'k.')
        plt.plot(points, fitvals.T[row], 'r')
        plt.show()

    # Restore the ordering & shape of the input array:
    fitvals = fitvals.reshape(tmpshape)
    fitvals = np.rollaxis(fitvals, 0, axis + 1)

    return fitvals
