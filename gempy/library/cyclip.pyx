# If switching to new versions of Python under anaconda,
# you may need to run this command again under the new
# environment.
# cythonize -a -i cyclip.pyx

# Specify that this source is nominally based on Python 3 syntax (though the
# code below is actually 2-vs-3 agnostic), to avoid a warning with v0.29+:
# cython: language_level=3

import numpy as np
from libc.math cimport sqrt
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef float median(float data[], unsigned short mask[], int has_mask,
                  int data_size):
    cdef float tmp[10000]
    cdef float x, y, med=0.
    cdef int i, j, k, l, m, ncycles, cycle, nused=0

    if has_mask:
        for i in range(data_size):
            if mask[i] == 0:
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
    return med

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void mask_stats(float data[], unsigned short mask[], int has_mask,
                int data_size, int return_median, double result[2]):
    cdef double mean, sum = 0., sumsq = 0., sumall = 0., sumsqall=0.
    cdef int i, nused = 0
    for i in range(data_size):
        sumall += data[i]
        sumsqall += data[i] * data[i]
        if has_mask and mask[i] == 0:
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
    cdef long i, ngood = 0
    for i in range(data_size):
        if mask[i] == 0:
            ngood += 1
    return ngood

@cython.boundscheck(False)
@cython.wraparound(False)
def iterclip(float [:] data, unsigned short [:] mask, float [:] variance,
             int has_var, int num_img, long data_size, double lsigma, double hsigma,
             int max_iters, int mclip, int sigclip):

    cdef long i, n, ngood, new_ngood
    cdef int iter, return_median
    cdef double result[2]
    cdef double avg, var, std
    cdef float low_limit, high_limit
    cdef float tmpdata[10000]
    cdef unsigned short tmpmask[10000]

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
                        tmpmask[n] |= 1
            else:
                for n in range(num_img):
                    std = sqrt(variance[n*data_size+i])
                    if tmpdata[n] < avg-lsigma*std or tmpdata[n] > avg+hsigma*std:
                        tmpmask[n] |= 1

            new_ngood = num_good(tmpmask, num_img)
            if new_ngood == ngood:
                break
            if not mclip:
                return_median = 0
            ngood = new_ngood
            iter += 1
        for n in range(num_img):
            mask[n*data_size+i] = tmpmask[n]

    return np.asarray(data), np.asarray(mask), np.asarray(variance)
