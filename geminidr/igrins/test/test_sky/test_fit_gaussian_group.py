import numpy as np
import pandas as pd

from numpy.typing import ArrayLike

def _gauss0_w_dcenters(xx, params, lines):
    """ Returns a gaussian function with the given parameters"""
    shift, sigma, height, background = params

    y_models = []
    with np.errstate(divide="ignore"):
        for line in lines:
            y_models.append(np.exp(-(((xx - (line + shift))/sigma)**2*0.5)))

    return height*np.array(y_models).sum(axis=0) + background


# def _gauss_w_dcenters_chi2(xx, yy, params, lines):
#     return np.sum((yy - _gauss0_w_dcenters(xx, params, lines))**2)


FIT_FAIL_RETURN_VALUE = [np.nan] * 4, None, None


def prepare_gaussian_group(x: np.ndarray, s: np.ndarray, lines: np.ndarray,
                           sigma_init=1.5,
                           max_sigma_scale=2,
                           fitrange_scale=2.5):
    """
    Prepare the fit that fits the spectrum with a group of gaussian that shares same offset and width.
    It returns sliced spectrum, fitting function, initial and bounds of the fit parameters.

    lines : initial x-coordinate of lines to fit
    sigma_init : initial sigma. A single value is given which will be shared with multiple lines. 
    """

    lines = np.array(lines)
    lines.sort()

    if not np.all(np.isfinite(lines)):  # if any of the position has nan
        return FIT_FAIL_RETURN_VALUE

    max_sigma = max_sigma_scale * sigma_init

    # The spectrum will be sliced and fit. We calculate the boundary of the slice.
    # The slice need to be larger than the bounds of x parameter.

    # The shift of x is bound to [-fitrange_scale, +fitrange_scale] * max_sigma.
    # The slice will be as large as this padded with addtional 2*max_sigma.

    xshift_max = fitrange_scale * max_sigma
    xmin = lines[0] - xshift_max - 2 * max_sigma
    xmax = lines[-1] + xshift_max + 2 * max_sigma

    # find the slice
    imin, imax = np.clip(np.searchsorted(x, [xmin, xmax]), 0, len(x))

    if imax - imin < 3:
        return FIT_FAIL_RETURN_VALUE

    sl = slice(imin, imax)

    xx = x[sl]
    yy = s[sl]

    # initial estimation of the height
    ymin, ymax = yy.min(), yy.max()
    yheight = ymax - ymin

    # dcenters0 = lines - lines[0]

    def _gauss(params, xx=xx, lines=lines):
        # return _gauss_w_dcenters_chi2(xx, yy, params, lines)
        return _gauss0_w_dcenters(xx, params, lines)

    # initial parameter and bounds
    params_ = [(0, (-xshift_max, xshift_max)),
               (sigma_init, (0, max_sigma)),
               (yheight, (0, 2*yheight)),
               (ymin, (ymin, ymax)) # baseline
               ]

    params0 = np.array([p for p, _ in params_])
    param_bounds = np.array([b for _, b in params_])

    return xx, yy, _gauss, params0, param_bounds


def fit_gaussian_group(x: np.ndarray, s: np.ndarray,
                       lines: np.ndarray,
                       sigma_init=1.5,
                       max_sigma_scale=2,
                       # drange_scale=5,
                       fitrange_scale=2.5):
    """
    Fit the spectrum with a group of gaussian that shares same offset and width.


    Parameters
    ----------
    x : x
    s : spectrum
    lines : initial x-coordinate of lines to fit
    sigma_init : initial sigma. A single value is given which will be
        shared with multiple lines. 

    Returns
    -------
    x : ndarray
        The solution. (offset, sigma, height, baseline)
    nfeval : int
        The number of function evaluations.
    rc : int
        Return code from the fitter, scipy.fmin_tnc


    """


    xx, yy, _gauss, params0, param_bounds = prepare_gaussian_group(x, s, lines,
                                                                   sigma_init=sigma_init,
                                                                   max_sigma_scale=max_sigma_scale,
                                                                   fitrange_scale=fitrange_scale)



    from scipy.optimize import fmin_tnc

    def chi2(params, gauss=_gauss, yy=yy):
        return np.sum((yy - gauss(params))**2)

    try:
        sol_ = fmin_tnc(chi2, params0,
                        bounds=param_bounds,
                        approx_grad=True, disp=0,
                        )
    except ValueError:
        raise

    return sol_


def main():
    import json
    j = json.load(open("test_data_for_fit_ohlines.json"))
    s = j["s"]
    df_ref_line = pd.DataFrame(j["ref_line"])

    grouped = df_ref_line.groupby("group_id").get_group(7)


    s = np.asarray(s)
    x = np.arange(len(s))
    lines = grouped["pixel"].values


    xx, yy, _gauss, params0, param_bounds = prepare_gaussian_group(x, s, lines)

    def chi2(params, gauss=_gauss, yy=yy):
        return np.sum((yy - gauss(params))**2)

    from scipy.optimize import fmin_tnc
    sol_ = fmin_tnc(chi2, params0,
                    bounds=param_bounds,
                    approx_grad=True, disp=0,
                    )


    # draw the result
    solx = sol_[0][0] + lines # np.array(d_centers0)
    y0 = sol_[0][-1]
    height = sol_[0][2]
    width = sol_[0][1]

    fig, ax = plt.subplots(1, 1, num=1, clear=True)
    ax.plot(xx, yy)
    color_fit = "C1"
    ax.vlines(solx, ymin=y0, ymax=height + y0, color=color_fit, ls=":")
    # ax.vlines(lines, ymin=0, ymax=sol_[0][2])
    ax.hlines([y0 + 0.5*height]*len(lines),
              xmin=solx-width, xmax=solx+width, color=color_fit, ls=":")

    xoo = np.linspace(xx[0], xx[-1], 100)
    ax.plot(xoo, _gauss(sol_[0], xoo), color=color_fit)
