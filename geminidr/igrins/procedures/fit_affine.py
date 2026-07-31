import numpy as np
# from scipy.interpolate import interp1d

from matplotlib.transforms import Affine2D


def fit_affine(xy1, xy2):
    """
    affine transfrom from xy1 to xy2
    xy1 : list of (x, y)
    xy2 : list of (x, y)

    Simply using leastsquare
    """

    # xy1f_ = np.empty((3, len(xy1f)))
    # xy1f_[:2, :] = xy1f.T
    # xy1f_[2, :] = 1
    xy1f_ = np.empty((len(xy1), 3))
    xy1f_[:, :2] = xy1
    xy1f_[:, 2] = 1

    abcdef = np.linalg.lstsq(xy1f_, xy2, rcond=None)

    return np.ravel(abcdef[0])


def fit_affine_clip(xy1f, xy2f):
    sol = fit_affine(xy1f, xy2f)

    affine_tr = Affine2D.from_values(*sol)

    xy1f_tr = affine_tr.transform(xy1f)  # [:,0], xy1f[:,1])

    # mask and refit
    dx_ = xy1f_tr[:, 0] - xy2f[:, 0]
    mystd = dx_.std()
    mm = np.abs(dx_) < 3. * mystd
    # print("SHAPE", xy1f.shape, mm[0].shape)
    sol = fit_affine(xy1f[mm], xy2f[mm])

    affine_tr = Affine2D.from_values(*sol)

    return affine_tr, mm
