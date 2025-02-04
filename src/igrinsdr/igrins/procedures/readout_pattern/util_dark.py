import numpy as np
import pandas as pd

from .readout_pattern_guard import (
    remove_pattern_from_guard)

from .readout_pattern_helper import (
    apply_rp_2nd_phase,
    apply_rp_3rd_phase)


def make_guard_n_bg_subtracted_images(dlist, rpc_mode="guard", bias_mask=None,
                                      log=None):

    cube = np.array([remove_pattern_from_guard(d)
                     for d in dlist])

    if len(cube) < 5:
        if log:
            log.stdinfo("No background will be estimated, since at least 5 "
                        "input AstroData objects are required")

        bg = np.zeros_like(cube[0])
        cube1 = cube
    else:
        bg = np.median(cube, axis=0)
        cube1 = cube - bg

    if rpc_mode == "guard":
        return cube1

    # cube20 = np.array([apply_rp_2nd_phase(d1) for d1 in cube1])
    cube2 = [apply_rp_2nd_phase(d1, mask=bias_mask) for d1 in cube1]

    if rpc_mode == "level2":
        return cube2

    cube3 = [apply_rp_3rd_phase(d1) for d1 in cube2]

    if rpc_mode == "level3":
        return cube3

    hdu_list = [
                ("GUARD_REMOVED", cube1),
                ("ESTIMATED_BG", bg),
                ("LEVEL2_REMOVED", cube2),
                ("LEVEL3_REMOVED", cube3)]

    return hdu_list


def _get_per_amp_stat(cube, namp=32, threshold=100):
    r = {}

    ds = cube.reshape((namp, -1))

    msk_100 = np.abs(ds) > threshold

    r["count_gt_threshold"] = np.sum(msk_100, axis=1)

    r["stddev_lt_threshold"] = [np.std(ds1[~msk1])
                                for ds1, msk1 in zip(ds, msk_100)]

    return r


def estimate_amp_wise_noise(kdlist, filenames=None):

    if filenames is None:
        filenames = [f"{i:02d}" for i in range(len(kdlist))]

    # kl = ["DIRTY", "GUARD-REMOVED", "LEVEL2-REMOVED", "LEVEL3-REMOVED"]
    dl = []
    for k, cube in kdlist:
        for fn, c in zip(filenames, cube):
            qq = _get_per_amp_stat(np.array(c))

            ka = dict(filename=fn, level=k)

            _ = [dict(amp=i,
                      stddev_lt_threshold=q1,
                      count_gt_threshold=q2, **ka)
                 for i, (q1, q2) in enumerate(zip(qq["stddev_lt_threshold"],
                                                  qq["count_gt_threshold"]))]

            dl.extend(_)

    return pd.DataFrame(dl)
