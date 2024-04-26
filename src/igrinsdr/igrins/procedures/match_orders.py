import numpy as np
import scipy

from scipy.signal import correlate
from scipy.ndimage import median_filter

def get_filtered(s):
    # The spectra need to be normalized in some way to make it
    # insensitive to bright lines. For now, we normalize by the total flux.

    # with warnings.catch_warnings():
    with np.errstate(invalid="ignore"):

        s_ = s - median_filter(s, 55)
        s_norm = np.nanstd(s_)
        # s_norm = np.nansum(np.abs(s_))

        # s_norm = np.nanmax(s_)

        s_clip = np.clip(s_, -0.1*s_norm, s_norm)/s_norm

    return s_clip

def get_filtered_s_list(s_list):
    s_list_clip = [get_filtered(s) for s in s_list]

    return s_list_clip

def _find_matching_spectra(s0, s_list):
    # we first filter the spectra and do the cross-correlation to find a best
    # candidate.

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)

        cor_list = [correlate(s0, s, mode="same") for s in s_list]

    with warnings.catch_warnings():
        msg = r'All-NaN (slice|axis) encountered'
        warnings.filterwarnings('ignore', msg)

        cor_max_list = [np.nanmax(cor) for cor in cor_list]

    center_indx_dst = np.nanargmax(cor_max_list)

    return center_indx_dst

def _check_thresh(unique, counts, frac_thresh=0.5):
    i = np.argmax(counts)
    maxn = counts[i]
    v = unique[i]

    if float(maxn) / counts.sum() > frac_thresh:
        return v

    if maxn > 2:
        msk = counts > 1
        # unique_msk = unique[msk]
        counts_msk = counts[msk]

        if float(maxn) / counts_msk.sum() > 0.5:
            return v

    return None



def match_specs(s_list_src, s_list_dst, frac_thresh=0.3):
    """
    try to math orders of src and dst.

    frac_thresh: raise error if the fraction of the convered d_order is less than this value.
    """

    center_indx0 = int(len(s_list_src)/2)

    delta_indx_list = []
    dstep = 5

    s_list = get_filtered_s_list(s_list_dst)

    for di in range(-dstep, dstep+1):

        center_indx = center_indx0 + di
        s = s_list_src[center_indx]
        s0 = get_filtered(s)

        dst_indx = _find_matching_spectra(s0, s_list)

        delta_indx = center_indx - dst_indx

        delta_indx_list.append(delta_indx)

    unique, counts = np.unique(delta_indx_list, return_counts=True)
    delta_indx = _check_thresh(unique, counts, frac_thresh)

    if delta_indx is None:
        print(delta_indx_list)
        raise ValueError("Concensus is not made for matching oders"
                         " : {}"
                         .format(dict(zip(unique, counts))))

    return delta_indx

def match_orders(orders, s_list_src, s_list_dst, frac_thresh=0.3):
    do = match_specs(s_list_src, s_list_dst, frac_thresh=frac_thresh)
    center_indx0 = 0
    center_indx_dst0 = center_indx0 + do
    orders_dst = (np.arange(len(s_list_dst))
                  + orders[center_indx0] + center_indx_dst0)

    return do, orders_dst


