import numpy as np
import scipy.ndimage as NI

from .trace_flat import trace_flat_edges, table_to_poly
from .iter_order import iter_order
from .smooth_continuum import get_smooth_continuum


def get_smoothed_order_spec(s):
    s = np.array(s)
    s[:4] = np.nan
    s[-4:] = np.nan

    try:
        k1, k2 = np.nonzero(np.isfinite(s))[0][[0, -1]]
    except IndexError:
        s0 = np.empty_like(s)
        s0.fill(np.nan)
        return s0

    s1 = s[k1:k2+1]

    s0 = np.empty_like(s)
    s0.fill(np.nan)
    s0[k1:k2+1] = NI.median_filter(s1, 40)
    return s0

def get_order_boundary_indices(s1, s0=None):
    # x = np.arange(len(s))

    # select finite number only. This may happen when orders go out of
    # chip boundary.
    s1 = np.array(s1)
    # k1, k2 = np.nonzero(np.isfinite(s1))[0][[0, -1]]
    s1[:4] = np.nan
    s1[-4:] = np.nan

    with np.errstate(invalid="ignore"):
        nonzero_indices = np.nonzero(s1 > 0.05)[0]  # [[0, -1]]

    # return meaningless indices if non-zero spectra is too short
    with np.errstate(invalid="ignore"):
        if len(nonzero_indices) < 5:
            return 4, 4

    k1, k2 = nonzero_indices[[0, -1]]
    k1 = max(k1, 4)
    k2 = min(k2, 2047-4)
    s = s1[k1:k2+1]

    if s0 is None:
        s0 = get_smoothed_order_spec(s)
    else:
        s0 = s0[k1:k2+1]

    mm = s > max(s) * 0.05
    dd1, dd2 = np.nonzero(mm)[0][[0, -1]]

    # mask out absorption feature
    smooth_size = 20
    # s_s0 = s-s0
    # s_s0_std = s_s0[np.abs(s_s0) < 2.*s_s0.std()].std()

    # mmm = s_s0 > -3.*s_s0_std

    s1 = NI.gaussian_filter1d(s0[dd1:dd2], smooth_size, order=1)
    # x1 = x[dd1:dd2]

    # s1r = s1 # ni.median_filter(s1, 100)

    s1_std = s1.std()
    s1_std = s1[np.abs(s1) < 2.*s1_std].std()

    s1[np.abs(s1) < 2.*s1_std] = np.nan

    indx_center = int(len(s1)*.5)

    left_half = s1[:indx_center]
    if np.any(np.isfinite(left_half)):
        i1 = np.nanargmax(left_half)
        a_ = np.where(~np.isfinite(left_half[i1:]))[0]
        if len(a_):
            i1r = a_[0]
        else:
            i1r = 0
        i1 = dd1+i1+i1r  # +smooth_size
    else:
        i1 = dd1

    right_half = s1[indx_center:]
    if np.any(np.isfinite(right_half)):
        i2 = np.nanargmin(right_half)
        a_ = np.where(~np.isfinite(right_half[:i2]))[0]

        if len(a_):
            i2r = a_[-1]
        else:
            i2r = i2
        i2 = dd1+indx_center+i2r
    else:
        i2 = dd2

    return k1+i1, k1+i2



def get_initial_spectrum_for_flaton(d, mask, slitedge_polyfit):
    specs = []
    psum = []

    npix_thresh = 20 # if there is less than 20 pixels along the column direction, do not use.

    for o, sl, m in iter_order(slitedge_polyfit):
        mask_good_pixels = (m & (mask[sl] == 0))

        s = np.sum(mask_good_pixels, axis=0)
        psum.append(s)
        bad_columns = s < npix_thresh

        dn = np.ma.array(d[sl], mask=~mask_good_pixels).filled(np.nan)
        s = np.nanmedian(dn,
                         axis=0)
        s[bad_columns] = np.nan
        if np.sum(bad_columns) >= 2040:
            specs.append(None)
        else:
            specs.append(s)

    return specs


def get_normalize_spectrum_for_flaton(specs):

    s_list = [None if s is None else get_smoothed_order_spec(s) for s in specs]
    i1i2_list = [(4, 4) if s is None else get_order_boundary_indices(s, s0)
                 for s, s0 in zip(specs, s_list)]
    s2_list = [None if s is None else get_smooth_continuum(s) for s, (i1, i2)
               in zip(s_list, i1i2_list)]

    return s_list, i1i2_list, s2_list

