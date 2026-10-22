import numpy as np
import scipy.ndimage as ni

from .image_combine import image_median
from .readout_pattern import destripe_helper as dh


def _make_background_mask(dark1):
    # esimate threshold for the initial background destermination
    dark1G = ni.median_filter(dark1, [15, 1])
    dark1G_med, dark1G_std = np.median(dark1G), np.std(dark1G)

    f_candidate = [1., 1.5, 2., 4.]
    for f in f_candidate:
        th = dark1G_med + f * dark1G_std
        m = (dark1G > th)

        k = np.sum(m, axis=0, dtype="f") / m.shape[0]
        if k.max() < 0.6:
            break
    else:
        # logger.warning("No suitable background threshold is found")
        m = np.zeros_like(m, dtype=bool)
        f, th = np.inf, np.inf

    k = dict(bg_med=dark1G_med, bg_std=dark1G_std,
             threshold_factor=k, threshold=th)
    return m, k


def make_background_mask(data_list):

    # subtract p64 usin the guard columns
    data_list1 = [dh.sub_p64_from_guard(d) for d in data_list]
    dark1 = image_median(data_list1)

    m, k = _make_background_mask(dark1)

    return m, k


def make_background_mask_from_combined(combined):

    m, k = _make_background_mask(combined)

    return m, k
