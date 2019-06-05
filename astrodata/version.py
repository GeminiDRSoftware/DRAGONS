#!/usr/bin/env python
"""
Holds the DRAGONS version to be propagated throught all the DRAGONS package
and to be used in the documentation.
"""

# --- Setup Version Here ---
API = 2
FEATURE = 1
BUG = 1
TAG = ''
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
        _version = "{:d}.{:d}".format(API, FEATURE)

    else:
        _tag = '-{:s}'.format(TAG) if len(TAG) > 0 else ''
        _version = "{:d}.{:d}.{:d}".format(API, FEATURE, BUG) + _tag

    return _version
