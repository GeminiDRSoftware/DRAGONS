import copy
import numpy as np
from astropy.convolution import convolve
from scipy.ndimage import median_filter


def ring_filter(nd, inner_radius, outer_radius, max_iters=1, inplace=False,
                replace_flags=65535, replace_func='median'):
    """Replace masked values with a median or mean of surrounding values.

    Parameters
    ----------
    nd : NDData
        The input NDData object. Will be modified in-place.
    inner_radius, outer_radius : float
        Inner and outer radius of the ring.
    max_iters : int
        Maximum number of iterations. If not reached the filtering is repeated
        until there are no masked values.
    replace_flags : int
        The flags to replace.
    replace_func : {'median', 'mean'}
        The function to compute the replacing value.

    """
    if replace_func not in ('median', 'mean'):
        raise ValueError('invalid value for replace_func')

    ndim = len(nd.shape)
    size = int(outer_radius)
    mgrid = np.indices([size*2 + 1] * ndim) - size
    mgrid *= mgrid
    footprint = np.sqrt(np.sum(mgrid, axis=0))
    footprint = ((footprint >= inner_radius) &
                 (footprint <= outer_radius)).astype(int)

    mask = (nd.mask & replace_flags) > 0
    filtered_data = nd.data
    iter = 0

    while iter < max_iters and np.any(mask):
        iter += 1
        if replace_func == "median":
            median_data = median_filter(filtered_data, footprint=footprint)
            filtered_data = np.where(mask, median_data, filtered_data)
            # If we're median filtering, we can update the mask...
            # if more than half the input pixels were bad, the
            # output is still bad.
            if iter < max_iters:
                mask = median_filter(mask, footprint=footprint)
        else:
            # "Mean" filtering is just convolution. The astropy
            # version handles the mask.
            median_data = convolve(filtered_data, footprint,
                                   mask=mask, boundary="extend")
            filtered_data = np.where(mask, median_data, filtered_data)
            # Output pixels are only bad if *all* the pixels in
            # the kernel were bad.
            if iter < max_iters:
                mask = np.where(convolve(mask, footprint,
                                boundary="extend") > 0.9999, True, False)

    out = nd if inplace else copy.deepcopy(nd)
    out.data[:] = filtered_data
    return out
