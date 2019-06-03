#!/usr/bin/env python

# --- Setup Version Here ---
api = 2
feature = 1
bug = 1
tag = ''
# --------------------------


def version(short=False):
    """
    Returns DRAGONS's version based on the api,
    feature and bug numbers.

    Returns
    -------
    str : formatted version
    """
    if short:
        v = "{:d}.{:d}".format(api, feature)

    else:
        t = '-{:s}'.format(tag) if len(tag) > 0 else ''
        v = "{:d}.{:d}.{:d}".format(api, feature, bug) + t

    return v
