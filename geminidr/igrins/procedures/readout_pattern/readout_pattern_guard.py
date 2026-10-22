from collections import OrderedDict
import numpy as np

from .readout_pattern import pipes


def get_column_percentile(guards, percentiles=None):
    if percentiles is None:
        percentiles = [10, 90]
    # guards = d[:, [0, 1, 2, 3, -4, -3, -2, -1]]
    r = OrderedDict(zip(percentiles, np.percentile(guards, percentiles)))
    r["std"] = np.std(guards[(r[10] < guards) & (guards < r[90])])
    return r


def get_guard_column_pattern(d, pattern_noise_recipes=None):
    if pattern_noise_recipes is None:
        pipenames_dark1 = ['amp_wise_bias_r2', 'p64_0th_order']
    else:
        pipenames_dark1 = pattern_noise_recipes

    guards = d[:, [0, 1, 2, 3, -4, -3, -2, -1]]

    pp = OrderedDict()
    for k in pipenames_dark1:
        p = pipes[k]
        _ = p.get(guards)
        guards = guards - p.broadcast(guards, _)

        guards = guards - np.median(guards)
        s = get_column_percentile(guards)
        pp[k] = dict(pattern=_, stat=s)

    return guards, pp

# from igrins.procedures.readout_pattern import pipes, apply as apply_pipe
# pipes_dark1 = [pipes[k] for k in list(pipes)[:2]]
# pipes_dark2 = [pipes[k] for k in list(pipes)[2:3]]


def remove_pattern_from_guard(d, recipes=None):

    if recipes is None:
        recipes = ['amp_wise_bias_r2', 'p64_0th_order']

    guards = d[:, [0, 1, 2, 3, -4, -3, -2, -1]]

    for k in recipes:
        p = pipes[k]
        _ = p.get(guards)
        d = d - p.broadcast(d, _)

    return d
