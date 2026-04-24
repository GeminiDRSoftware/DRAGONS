import numpy as np
# import matplotlib.pyplot as plt

from .destriper import stack64, concat


def _get_ny_slice(ny, dy):
    n_dy = ny//dy
    dy_slices = [slice(iy*dy, (iy+1)*dy) for iy in range(n_dy)]
    return dy_slices


def get_pattern64_from_each_column(d):
    ds = stack64(d)
    p = concat(ds, [1, -1], 16)
    return ds, p


def get_median_guard_row(d):
    return np.median(d[[0, 1, 2, 3, -4, -3, -2, -1], :], axis=0)


def get_median_guard_column(d):
    return np.median(d[:, [0, 1, 2, 3, -4, -3, -2, -1]], axis=1)


def get_pattern64_from_guard_column(d):
    dd = np.median(d[:, [0, 1, 2, 3, -4, -3, -2, -1]], axis=1)
    ds, p = get_pattern64_from_each_column(dd)
    return ds, p


def subtract_column_median(d):
    return d - np.median(d, axis=0)


def subtract_row_median(d):
    return d - np.median(d, axis=1)[:, np.newaxis]


def get_row64_median(d):
    ny_slices = _get_ny_slice(2048, 64)
    return np.concatenate([np.broadcast_to(np.median(d[sl]), (len(d[sl]), ))
                           for sl in ny_slices])


def subtract_row64_median(d):
    ny_slices = _get_ny_slice(2048, 64)
    return np.concatenate([d[sl] - np.median(d[sl]) for sl in ny_slices])


def sub_row(d, r):
    return d - r


def sub_column(d, c):
    return d - c[:, np.newaxis]


# sky?
def sub_guard_column(d):
    k = get_median_guard_column(d)
    k0 = subtract_row64_median(k)
    ds, p = get_pattern64_from_each_column(k0)
    vv = d - p[:, np.newaxis]
    xx = np.median(vv[-512:], axis=0)
    return vv - xx


def sub_p64(d):

    k0 = get_row64_median(d)
    dk = d - k0[:, np.newaxis]
    ds, p = get_pattern64_from_each_column(dk)
    d0k = subtract_row_median(dk - p)

    return d0k


# try subtract row64 first.

def sub_p64_from_guard(d):
    """ This removes p64 esimated from the guard column. You may need to further
remove median row/column values, but proper background needs to be removed first.
    """

    k = get_median_guard_column(d)
    k0 = subtract_row64_median(k)
    ds, p = get_pattern64_from_each_column(k0)
    dd = d - p[:, np.newaxis]
    # d2 = subtract_row_median(subtract_column_median(dd))

    return dd


def get_p64_mask(d, destrip_mask):
    # k = get_median_guard_column(d)
    # k0 = get_row64_median(k)
    # dk = d - k0[:, np.newaxis]

    # dk_mskd = np.ma.array(d, mask=destrip_mask)  # .filled(np.nan)
    kd = stack64(d, destrip_mask)

    p = concat(kd, [1, -1], 16)

    return kd, p


def sub_p64_mask(d, destrip_mask):
    k = get_median_guard_column(d)
    k0 = get_row64_median(k)
    dk = d - k0[:, np.newaxis]

    dk_mskd = np.ma.array(dk, mask=destrip_mask)  # .filled(np.nan)
    kd = stack64(dk, destrip_mask)

    p = concat(kd, [1, -1], 16)

    # vertical pattern using the masked data
    d5p = dk_mskd.filled(np.nan) - p
    k0 = np.nanmedian(d5p, axis=1)
    d5 = (dk - p) - k0[:, np.newaxis]
    return d5


def sub_p64_guard(v):
    k = get_median_guard_column(v)
    k0 = get_row64_median(k)
    ds, p = get_pattern64_from_each_column(k - k0)
    vv = v - k0[:, np.newaxis] - p[:, np.newaxis]
    return vv
    # d2 = subtract_row_median(subtract_column_median(dd))


def sub_bg64_from_guard(v):
    k = get_median_guard_column(v)
    k0 = get_row64_median(k)
    v1 = v - k0[:, np.newaxis]
    return v1


def sub_p64_upper_quater(v, subtract_bg64=False):
    if subtract_bg64:
        # It seems that this usually worsen the background
        v1 = sub_bg64_from_guard(v)
    else:
        v1 = v
    msk = np.ones(v.shape, dtype=bool)
    msk[2048 - 384:] = False
    td, p = get_p64_mask(v1, msk)

    return v1 - p


def sub_p64_with_bg(d, bg):
    ds, p = get_pattern64_from_each_column(d - bg)
    return d - p


def sub_p64_with_mask(d0, destrip_mask=None):
    if destrip_mask is not None:
        mask = ~np.isfinite(d0) | destrip_mask
    else:
        mask = ~np.isfinite(d0)

    ds, p = get_p64_mask(d0, mask)
    return d0 - p

