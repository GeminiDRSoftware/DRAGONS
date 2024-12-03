import numpy as np
import scipy.spatial as spatial

from itertools import repeat

from .fit_gaussian import fit_gaussian_simple
from .find_peak import find_peaks


def reidentify(s, x_list, x=None, sigma_init=1.5):
    """
    given spectrum s, try to reidentify lines as x_list as initial point.

    x_list: list of list of x values. e.g., [[1, 2], [4, 6]]. Each list elements shares a constant offset value.
    sigma_init : a single value, or an iterator of single value.
    """

    if x is None:
        x = np.arange(len(s))

    try:
        sigma_iter = iter(sigma_init)
    except TypeError:
        sigma_iter = repeat(sigma_init)

    fit_results = []

    for lines_pixel, sigma in zip(x_list, sigma_iter):

        sol_ = fit_gaussian_simple(x, s, lines_pixel,
                                   sigma_init=sigma,
                                   do_plot=False)

        # if fit result is not good (for example, initial position of the line is off too much),
        # we redo the file by increasing the fit range.
        if sol_[-1] not in [0, 1, 2, 3]:
            sol_ = fit_gaussian_simple(x, s, lines_pixel,
                                       sigma_init=sigma,
                                       drange_scale=10,
                                       do_plot=False)
            if sol_[-1 not in [0, 1, 2, 3]]:
                sol_[0][0] = np.nan

        fit_results.append(sol_)

    return fit_results


def reidentify2(s, ref_x_list, sigma=3):
    """
    given spectrum s, identify line features from s. Find nearest match
    to x_list.
    x_list is transformed with trans_func if not None.
    """

    sol_list = find_peaks(s, sigma=sigma, ax=None)
    center_list = np.array([sol_[0] for sol_ in sol_list])

    kdtree = spatial.KDTree(center_list.reshape([-1,1]))


    ref_x_list = np.array(ref_x_list)
    dists, indices = kdtree.query(ref_x_list.reshape([-1,1]))

    # filter out multiple hits. Only the nearest one remains.
    import operator
    from itertools import groupby, count

    identified_lines = [None] * len(ref_x_list)
    for k, l in groupby(zip(dists, count(), indices),
                        operator.itemgetter(-1)):
        l = list(l)
        d, ref_i, i = min(l) #i = np.argmin([l1[-1] for l1 in l])
        identified_lines[ref_i] = (sol_list[i], d)

    return identified_lines

def reidentify_lines_all2(s_list_dst, ref_lines_list,
                          sol_list_transform):

    reidentified_lines = []
    for s, ref_lines, sol_tr in zip(s_list_dst,
                                    ref_lines_list,
                                    sol_list_transform["sol_list"]):

        ref_x_list_ = [s_[1][0] for s_ in ref_lines]
        ref_x_list = sol_tr(np.array(ref_x_list_))

        lines = reidentify2(s, ref_x_list, sigma=3)

        reidentified_lines.append(lines)

    if 0:
        dd = np.concatenate([[l1[-1] for l1 in lines_ if l1] \
                             for lines_ in reidentified_lines])

        hist(dd, bins=np.linspace(0, 4, 20))

    # make masks for distant matches

    msk_list = []
    for ref_lines, lines in zip(ref_lines_list,
                                reidentified_lines):
        # if l1 is not None and matched distance(l1[-1]) is small enough
        msk = [(l1 and l1[-1] < 2.) for l1 in lines]
        msk_list.append(msk)

    # compress and convert to json-compativel types
    from itertools import compress
    reidentified_lines_with_id = []
    for ref_lines, lines, msk in zip(ref_lines_list,
                                     reidentified_lines,
                                     msk_list):
        id_list = list(id_ for id_, d_ in compress(ref_lines, msk))
        fit_list = list(list(d_[0]) for d_ in compress(lines, msk))
        reidentified_lines_with_id.append((id_list, fit_list))

    return reidentified_lines_with_id



def reidentify_lines_all(s_list, ref_positions_list,
                         sol_list_transform=None):

    x = np.arange(2048)
    fit_results = []

    for ref_positions, s in zip(ref_positions_list, s_list):

        results_ = reidentify(s, ref_positions, x=x, sigma_init=1.5)

        dpix_list = [p_ - p_[0] for p_ in ref_positions]

        fit_results.append((results_, dpix_list))

    return fit_results
