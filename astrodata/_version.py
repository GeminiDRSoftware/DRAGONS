#!/usr/bin/env python
"""
Holds the DRAGONS version to be propagated throught all the DRAGONS package
and to be used in the documentation.
"""

# --- Setup Version Here ---
API = 4
FEATURE = 1
BUG = 0
TAG = 'dev'


def version(short=False, tag=TAG):
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
        _tag = '_{:s}'.format(tag) if tag else ''
        _version = "{:d}.{:d}.{:d}".format(API, FEATURE, BUG) + _tag

    return _version
