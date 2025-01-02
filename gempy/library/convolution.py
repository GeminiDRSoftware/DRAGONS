import numpy as np

from scipy.interpolate import make_interp_spline

from gempy.library import astrotools as at

"""
Various convolution functions should go here. They can have as many parameters
as you like, but the "line_spread_function" that is passed must be compatible
with the definition in the convolve() function, i.e., take 2 parameters: a
wavelength and an array of wavelength offsets
"""


def gaussian_constant_r(w0, dw, r):
    """A Gaussian with FWHM = w0 / r"""
    sigma = 0.42466 * w0 / r
    z = np.exp(-0.5 * (dw / sigma) ** 2)
    return z / z.sum()


def boxcar(w0, dw, width):
    """A boxcar of full-width width nm"""
    z = abs(dw) <= 0.5 * width
    return z / z.sum()


def convolve(w, y, func, dw, sampling=1):
    """
    Convolve a spectrum with a line spread function that is itself a function
    of wavelength. There are various assumptions here, e.g. the wavelength
    scale is approximately linear across the size of the kernel, that may need
    to be revisited later.

    Parameters
    ----------
    w: array
        wavelengths
    y: array
        data
    func: callable
        function that defines the convolution kernel representing the line
        spread function. This must take 2 arguments: the central wavelength
        and an array of wavelength offsets. The returned kernel must be
        normalized (sum=1).
    dw: float
        half-extent of kernel (in nm)
    sampling: int
        to speed up the calculation, the kernel can be calculated at discrete
        intervals and then interpolated to get its form at other locations

    Returns
    -------
    array: the convolved array, of the same size as x
    """
    #start = datetime.now()
    # Calculate size of kernel in pixels
    diffs = np.diff(w)
    rev = any(diffs < 0)
    if rev and not all(diffs < 0):
        raise ValueError("Wavelength array is not monotonic")
    i = int(dw / abs(diffs).min() + 1)

    kernel_size = i + i + 1
    yout = np.zeros_like(y)
    nw = y.shape[-1]
    if sampling == 1:  # this is a bit quicker
        for j in range(0, nw):
            start = max(j-i, 0)
            yout[..., start:j+i+1] = (yout[..., start:j+i+1] +
                                      np.outer(y.T[j], func(w[j], w[start:j+i+1] - w[j])))
    else:
        ww = w[:nw:sampling]
        yconv = np.zeros((ww.size, kernel_size))
        for jj, j in enumerate(range(0, nw, sampling)):
            start = max(j-i, 0)
            yconv[jj, start-j+i:] = func(w[j], w[start:j+i+1]-w[j])
        for j in range(kernel_size):
            z = y * np.interp(w, ww, yconv[:, j])
            yout[..., j:j+nw-i] += z[..., i:nw+i-j]

    #print(datetime.now() - start)
    return yout


def resample(wout, w, data):
    """
    Resample a spectrum to a different output wavelength array. This is done
    by interpolating the data with a cubic spline and integrating this between
    the edges of the pixels. There will be uncertain (possibly fatal) edge
    effects if the input array does not fully cover the wavelength regime of
    the output array.

    Parameters
    ----------
    wout: array
        output wavelength array
    w: array
        input wavelength array (*must* be in increasing order)
    data: array
        data to be resampled

    Returns
    -------
    resampled array of same size as wout
    """
    diffs = np.diff(w)
    rev_in = any(diffs < 0)
    if rev_in and not all(diffs < 0):
        raise ValueError("Input wavelength array is not monotonic")
    diffs = np.diff(wout)
    rev_out = any(diffs < 0)
    if rev_out and not all(diffs < 0):
        raise ValueError("Output wavelength array is not monotonic")

    _slice = slice(None, None, -1) if rev_in else slice(None)
    spline = make_interp_spline(w[_slice], data[_slice], axis=-1, k=3)
    spline.extrapolate = False
    edges = at.calculate_pixel_edges(wout)
    int_spline = spline.antiderivative()(edges)
    assert np.all(np.isfinite(int_spline)), "Error in resampling"
    if rev_out:
        return -np.diff(int_spline) / abs(np.diff(edges))
    return np.diff(int_spline) / abs(np.diff(edges))
