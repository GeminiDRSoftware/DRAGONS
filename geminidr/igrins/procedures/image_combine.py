
import numpy as np

def image_median(dd, badmasks=None):
    if badmasks is None:
        ddm = np.median(dd, axis=0)
    else:
        ddm = np.nanmedian(np.ma.array(dd, mask=badmasks).filled(np.nan), axis=0)
    return ddm
