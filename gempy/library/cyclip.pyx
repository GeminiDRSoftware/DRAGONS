# Copyright(c) 2018-2019 Association of Universities for Research in Astronomy, Inc.

# Specify that this source is nominally based on Python 3 syntax
# (though the code below is actually 2-vs-3 agnostic), to avoid a warning with
# v0.29+:
#
# cython: language_level=3
#

"""
If switching to new versions of Python under anaconda, you may need to run
this command again under the new environment.::

    $ cythonize -i cyclip.pyx

"""

import numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free


@cython.boundscheck(False)
@cython.wraparound(False)
cdef float median(float data[], unsigned short mask[], int has_mask,
                  int data_size) except? -1:
    """
    One-dimensional true median, with optional masking.

    Parameters
    ----------
    data : float array
        (add description)
    mask : unsigned short array
        (add description)
    has_mask : int
        (add description)
    data_size : int
        (add description)

    Returns
    -------
    med : (add type)
        (add description)
    """
    cdef float x, y, med=0.
    cdef int i, j, k, l, m, ncycles, cycle, nused=0
    cdef float *tmp = <float *> malloc(data_size * sizeof(float))

    if not tmp:
        raise MemoryError()

    if has_mask:
        for i in range(data_size):
            if mask[i] < 65535:
                tmp[nused] = data[i]
                nused += 1
    if nused == 0:  # if not(has_mask) or all(mask)
        for i in range(data_size):
            tmp[i] = data[i]
        nused = data_size

    ncycles = 2 - nused % 2
    for cycle in range(0, ncycles):
        k = (nused - 1) // 2 + cycle
        l = 0
        m = nused - 1
        while (l < m):
            x = tmp[k]
            i = l
            j = m
            while True:
                while (tmp[i] < x):
                    i += 1
                while (x < tmp[j]):
                    j -= 1
                if i <= j:
                    y = tmp[i]
                    tmp[i] = tmp[j]
                    tmp[j] = y
                    i += 1
                    j -= 1
                if i > j:
                    break
            if j < k:
                l = i
            if k < i:
                m = j
        if cycle == 0:
            med = tmp[k]
        else:
            med = 0.5 * (med + tmp[k])

    free(tmp)

    return med


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void mask_stats(float data[], unsigned short mask[], int has_mask,
                long data_size, int return_median, double result[2]):
    """
    Returns either the mean or median (and variance) of the unmasked pixels in
    an array.

    Parameters
    ----------
    data : float array
        1D array of datapoints
    mask : unsigned short array
        1D array indicating which pixels are masked (non-zero)
    has_mask : int
        is there a mask?
    data_size : long
        length of array
    return_median : int
        return median?

    Returns
    -------

    """
    cdef double mean, sum = 0., sumsq = 0., sumall = 0., sumsqall=0.
    cdef int i, nused = 0
    for i in range(data_size):
        sumall += data[i]
        sumsqall += data[i] * data[i]
        if has_mask and mask[i] < 32768:
            sum += data[i]
            sumsq += data[i] * data[i]
            nused += 1
    if nused == 0:
        sum = sumall
        sumsq = sumsqall
        nused = data_size
    mean = sum / float(nused)
    if return_median:
        result[0] = <double>median(data, mask, has_mask, data_size)
    else:
        result[0] = mean
    result[1] = sumsq / nused - mean*mean


cdef long num_good(unsigned short mask[], long data_size):
    """
    Returns the number of unmasked pixels in an array.

    Parameters
    ----------
    mask : unsigned short array
        (add description)
    data_size : long
        (add description)

    Returns
    -------
    ngood : (?)
        (add description)
    """
    cdef long i, ngood = 0

    for i in range(data_size):
        if mask[i] < 32768:
            ngood += 1

    return ngood


@cython.boundscheck(False)
@cython.wraparound(False)
def iterclip(float [:] data, unsigned short [:] mask, float [:] variance,
             int has_var, int num_img, long data_size, double lsigma,
             double hsigma, int max_iters, int mclip, int sigclip):
    """
    Iterative sigma-clipping. This is the function that interfaces with python.

    Parameters
    ----------
    data : float array
        1D arrays of input, each made up of num_img points for each input pixel.
    mask : unsigned short array
        (add description)
    variance : float array
        (add description)
    has_var : int
        Worry about the input variance array?
    num_img : int
        Number of input images.
    data_size : long
        Number of pixels per input image
    lsigma : double
        Number of standard deviations for clipping below the mean.
    hsigma : double
        Number of standard deviations for clipping above the mean.
    max_iters : int
        Maximum number of iterations to compute
    mclip : int
        Clip around the median rather than mean?
    sigclip : int
        Perform sigma-clipping using the pixel-to-pixel scatter, rather than
        use the variance array?

    Returns
    -------
    data : numpy.ndarray(np.float)
        (add description)
    mask : numpy.ndarray(np.float)
        (add description)
    variance : numpy.ndarray(np.float)
        (add description)

    """

    cdef long i, n, ngood, new_ngood
    cdef int iter, return_median
    cdef double result[2]
    cdef double avg, var, std
    cdef float low_limit, high_limit

    cdef float *tmpdata = <float *> malloc(num_img * sizeof(float))
    if not tmpdata:
        raise MemoryError()

    cdef unsigned short *tmpmask = <unsigned short *> malloc(num_img * sizeof(unsigned short))
    if not tmpmask:
        raise MemoryError()

    if max_iters == 0:
        max_iters = 100

    for i in range(data_size):
        iter = 0
        return_median = 1
        for n in range(num_img):
            tmpdata[n] = data[n*data_size+i]
            tmpmask[n] = mask[n*data_size+i]
        ngood = num_good(tmpmask, num_img)
        while iter < max_iters:
            mask_stats(tmpdata, tmpmask, 1, num_img, return_median, result)
            avg = result[0]
            if has_var == 0 or sigclip:
                std = sqrt(result[1])
                low_limit = avg - lsigma * std
                high_limit = avg + hsigma * std
                for n in range(num_img):
                    if tmpdata[n] < low_limit or tmpdata[n] > high_limit:
                        tmpmask[n] |= 32768
            else:
                for n in range(num_img):
                    std = sqrt(variance[n*data_size+i])
                    if tmpdata[n] < avg-lsigma*std or tmpdata[n] > avg+hsigma*std:
                        tmpmask[n] |= 32768

            new_ngood = num_good(tmpmask, num_img)
            if new_ngood == ngood:
                break
            if not mclip:
                return_median = 0
            ngood = new_ngood
            iter += 1
        for n in range(num_img):
            mask[n*data_size+i] = tmpmask[n]

    free(tmpdata)
    free(tmpmask)

    return np.asarray(data), np.asarray(mask), np.asarray(variance)
