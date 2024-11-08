import numpy as np


def getsigma(self, adinputs, **params):
    try:
        dark = self.streams['darks'][0]
    except (KeyError, TypeError, IndexError):
        raise OSError("A SET OF DARKS IS REQUIRED INPUT")
    for dark_ext in dark:
        d = dark_ext.data
        d_std = np.nanstd(d)
        d_std2 = d[np.abs(d) < d_std * 3].std()
        print(d_std2)
        dark_ext.hdr["BG_STD"] = d_std2
    print(dark_ext.hdr["BG_STD"])
    return d_std2

def badpixel_mask(d,
                  sigma_clip1=100, sigma_clip2=10,
                  medfilter_size=None, d_std2=None,):

    """
    msk1 : sigma_clip1
    """
    import scipy.ndimage as ni

    if d_std2 is None:
        d_std = np.nanstd(d)#.std()
        d_std2 = d[np.abs(d) < d_std * 3].std()

    msk1_ = d > d_std2 * sigma_clip1

    msk1 = ni.convolve(msk1_, [[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])

    if medfilter_size is not None:
        d_med = ni.median_filter(d, size=medfilter_size)
        d = d - d_med

    msk2 = np.abs(d) > d_std2 * sigma_clip2

    msk = msk1 | msk2

    return d_std2, msk

