from functools import partial

import numpy as np
from datetime import datetime

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
    """A boxcar of full width width nm"""
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
    if any(diffs <= 0):
        raise ValueError("Wavelength array must be monotonically increasing")
    i = int(dw / diffs.min() + 1)

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
    Resample a spectrum to a different output wavelength array. This follows
    the method of nddops.sum1d() in DRAGONS by taking fractional pixels. There
    will be uncertain (possibly fatal) edge effects if the input array does
    not fully cover the wavelength regime of the output array.

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
    dw_in = np.diff(w)
    if any(dw_in < 0):
        raise ValueError("Wavelength array must be monotonically increasing")

    if len(data.shape) > 1:
        output = np.empty((data.shape[0], wout.size), dtype=data.dtype)
    else:
        output = np.empty_like(wout, dtype=data.dtype)
    indices = np.arange(w.size)
    edges = np.r_[[1.5 * wout[0] - 0.5 * wout[1]],
                  np.array([wout[:-1], wout[1:]]).mean(axis=0),
                  [1.5 * wout[-1] - 0.5 * wout[-2]]]
    dw_out = abs(np.diff(edges))

    # have to do some flipping if wavelengths are in descending order
    if wout[1] > wout[0]:
        xedges = np.interp(edges, w, indices)
    else:
        xedges = np.interp(edges[::-1], w, indices)
    for i in range(wout.size):
        x1, x2 = xedges[i:i+2]
        ix1 = int(np.floor(x1 + 0.5))
        ix2 = int(np.ceil(x2 - 0.5))
        fx1 = ix1 - x1 + 0.5
        fx2 = x2 - ix2 + 0.5
        #output[..., i] = (fx1 * data[..., ix1] + data[..., ix1+1:ix2].sum(axis=-1) +
        #                  fx2 * data[..., ix2]) / (x2 - x1)
        output[..., i] = (fx1 * data[..., ix1] * dw_in[ix1] +
                          (data[..., ix1 + 1:ix2] * dw_in[ix1 + 1:ix2]).sum(axis=-1) +
                          fx2 * data[..., ix2] * dw_in[ix2]) / dw_out[i]
    return output if wout[1] > wout[0] else output[..., ::-1]

