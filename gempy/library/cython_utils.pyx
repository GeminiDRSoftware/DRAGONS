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

    $ cythonize -i cython_utils.pyx

"""
import numpy as np
cimport cython
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free


# These functions are used by nddops.py for combining images, especially
# in the stackFrames() primitive


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


@cython.exceptval(check=False)
cdef long num_good(unsigned short mask[], int data_size) except? 0:
    """
    Returns the number of unmasked pixels in an array.

    Parameters
    ----------
    mask : unsigned short array
        array of mask pixels
    data_size : long
        size of array

    Returns
    -------
    ngood : long
        number of good pixels in stack
    """
    cdef long i, ngood = 0

    for i in range(data_size):
        if mask[i] < 32768:
            ngood += 1

    return ngood


@cython.boundscheck(False)
@cython.wraparound(False)
def iterclip(float [:] data, unsigned short [:] mask, float [:] variance,
             int has_var, long num_img, long data_size, double lsigma,
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
    num_img : long
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


##############################################################
# The following function is used by the BruteLandscapeFitter()

def landstat(double [:] landscape, int [:] coords, int [:] len_axes,
             int num_axes, int num_coords):
    cdef int c, coord, i, j, l, ok
    cdef float sum=0.
    cdef int sum2=0

    for i in range(num_coords):
        c = i
        ok = 1
        l = 0
        for j in range(num_axes-1, -1, -1):
            coord = coords[c]
            if coord >=0 and coord < len_axes[j]:
                if j < num_axes - 1:
                    l += coord * len_axes[j+1]
                else:
                    l += coord
            else:
                ok = 0
            c += num_coords
        if ok:
            sum += landscape[l]

    return sum


##############################################################
# The following code is used by polynomial interpolators

@cython.boundscheck(False)
@cython.wraparound(False)
def polyinterp(float [:] array_in, int [:] inlengths, int ndim,
               float [:] array_out, int [:] outlengths,
               float [:] geomap, int affinity, int order, float cval):
    """
    Perform polynomial interpolation of one array to another array of the
    same dimensionality.

    Parameters
    ----------
    array_in : float array
        input array that is being transformed
    inlengths : int array
        axis lengths of input image (in python order)
    ndim : int
        number of dimensions of array
    array_out : float array
        pre-created output array for placing results
    outlengths : int array
        axis lengths of output image (in python order)
    geomap : float array
        either (affinity=False) ndim * npixout array representing locations
            in array_in where each pixel in array_out maps back to
        or (affinity=True) ndim * (ndim + 1) array of affine transformation
            matrix and offset
    affinity : int
        boolean to describe nature of geomap
    order : int (3 or 5)
        order of interpolation (will be downgraded near boundaries)
    cval : float
        value to insert for out-of-bounds pixels
    """
    cdef long i, nstep, npixout=1
    cdef int j, k, dim, totpix, num_pix=1
    cdef float p, q, pp, qq, ppp, qqq, coord, sum
    cdef float tmpcoeff[6]
    cdef int pix1, npix
    cdef int coords[10]

    for dim in range(ndim):
        num_pix *= order + 1
        coords[dim] = 0
        npixout *= outlengths[dim]
    coords[ndim] = 1

    # space for all coefficients
    cdef float *coeffs = <float *> malloc(num_pix * sizeof(float))
    if not coeffs:
        raise MemoryError()

    # space for pixel number indices
    cdef long *pixels = <long *> malloc(num_pix * sizeof(long))
    if not pixels:
        raise MemoryError()

    for i in range(npixout):
        for j in range(num_pix):
            coeffs[j] = 1.0
            pixels[j] = 0

        totpix = 1  # total number of input pixels contributing
        nstep = 1  # indexing increment in this dimension
        for dim in range(ndim - 1, -1, -1):
            if totpix > 0:
                if affinity:
                    coord = 0.
                    for j in range(ndim + 1):
                        coord += geomap[dim * (ndim + 1) + j] * coords[j]
                else:
                    coord = geomap[dim * npixout + i]
                if coord >= 0 and coord < inlengths[dim] - 1:
                    pix1 = int(coord)
                    p = coord - pix1
                    q = 1. - p
                    if pix1 > 0 and pix1 < inlengths[dim] - 2 and order >= 3:
                        pp = p * (p * p - 1.) / 6.
                        qq = q * (q * q - 1.) / 6.
                        if pix1 > 1 and pix1 < inlengths[dim] - 3 and order >= 5:
                            ppp = pp * 0.05 * (p * p - 4.)
                            qqq = qq * 0.05 * (q * q - 4.)
                            pix1 -= 2
                            npix = 6
                            tmpcoeff[0] = qqq
                            tmpcoeff[1] = qq - 4 * qqq + ppp
                            tmpcoeff[2] = q - qq - qq + pp + 6 * qqq - 4 * ppp
                            tmpcoeff[3] = p - pp - pp + qq + 6 * ppp - 4 * qqq
                            tmpcoeff[4] = pp - 4 * ppp + qqq
                            tmpcoeff[5] = ppp
                        else:  # cubic interpolation
                            pix1 -= 1
                            npix = 4
                            tmpcoeff[0] = qq
                            tmpcoeff[1] = q + pp - qq - qq
                            tmpcoeff[2] = p + qq - pp - pp
                            tmpcoeff[3] = pp
                    else:  # only linear interpolation
                        npix = 2
                        tmpcoeff[0] = q
                        tmpcoeff[1] = p

                    for j in range(totpix):
                        for k in range(npix - 1, -1, -1):
                            pixels[k * totpix + j] = pixels[j] + (pix1 + k) * nstep
                            coeffs[k * totpix + j] = coeffs[j] * tmpcoeff[k]

                    totpix *= npix
                    nstep *= inlengths[dim]

                else:
                    totpix = 0

        if totpix > 0:
            sum = 0.
            for j in range(totpix):
                sum += coeffs[j] * array_in[pixels[j]]
            array_out[i] = sum
        else:
            array_out[i] = cval

        coords[ndim - 1] += 1
        for j in range(ndim - 1, -1, -1):
            if coords[j] == outlengths[j]:
                coords[j] = 0
                if j > 0:
                    coords[j - 1] += 1


##############################################################
# The following function is used by the telluric code

@cython.boundscheck(False)
@cython.wraparound(False)
def get_unmasked(float [:] waves, float [:] x, int x_size, unsigned short [:] good):
    """
    Determine which elements of waves[] are in x[] and returns the good pixels.
    good[] should be set to an array of zeros before calling
    """
    cdef int i=0, j=0
    for j in range(x_size):
        if waves[i] == x[j]:
            good[i] = 1
        else:
            while waves[i] != x[j]:
                i += 1
            good[i] = 1
