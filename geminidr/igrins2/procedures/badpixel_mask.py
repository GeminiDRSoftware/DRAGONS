import numpy as np
import scipy.ndimage as ni

# from igrinsdr.igrins.procedures.trace_flat import get_flat_mask_auto
from .trace_flat import get_flat_mask_auto


# def getsigma(self, adinputs, **params):
#     try:
#         dark = self.streams['darks'][0]
#     except (KeyError, TypeError, IndexError):
#         raise OSError("A SET OF DARKS IS REQUIRED INPUT")
#     for dark_ext in dark:
#         d = dark_ext.data
#         d_std = np.nanstd(d)
#         d_std2 = d[np.abs(d) < d_std * 3].std()
#         print(d_std2)
#         dark_ext.hdr["BG_STD"] = d_std2
#     print(dark_ext.hdr["BG_STD"])
#     return d_std2


def make_igrins_hotpixel_mask(d,
                              sigma_clip1=100, sigma_clip2=10,
                              medfilter_size=None):

    """
    msk1 : sigma_clip1
    """

    d_std = np.nanstd(d)
    d_std = d[np.abs(d) < d_std * 3].std()

    msk1_ = d > d_std * sigma_clip1

    msk1 = ni.convolve(msk1_, [[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

    if medfilter_size is not None:
        d_med = ni.median_filter(d, size=medfilter_size)
        d = d - d_med

    msk2 = np.abs(d) > d_std * sigma_clip2

    msk = msk1 | msk2

    return d_std, msk


def make_igrins_deadpixel_mask(flat_on, flat_std, deadpix_thresh=0.6, smooth_size=9):
    # Given the stacked flat on image, we smooth it using smoothe_size and flag out pixels
    # whose ratio of flat_on / smoothed is lower than the threshold.
    # While this will give dead pixels in the slit illuminated regions, it will also give lots of
    # pixels in the un-illuminated region. This will be okay as these are will be flagged un-illuminated
    # anyway.
    # However, we try to remove these pixels using some masks.

    # original version of the code from plp calculates normalization factor and use normalized flat.
    # Upon reviewing the code, it was found that the normalization is not necessary, so we do not.

    flat_smoothed = ni.median_filter(flat_on,
                                     [smooth_size, smooth_size])
    flat_ratio = flat_on/flat_smoothed

    # To remove pixels in un-illuminated area, we will use two masks. One using the standard deviation
    # of a given pixel. Another one using the flat on image itself.

    flat_std = ni.median_filter(flat_std,
                                size=(3, 3))
    flat_std_mask = (flat_smoothed - flat_on) > 5*flat_std
    flat_mask = get_flat_mask_auto(flat_on)

    deadpix_mask = ((flat_ratio < deadpix_thresh) &
                    flat_std_mask & flat_mask)

    deadpix_mask[[0, 1, 2, 3, -4, -3, -2, -1]] = False
    deadpix_mask[:, [0, 1, 2, 3, -4, -3, -2, -1]] = False

    return deadpix_mask
