#!/usr/bin/env python
"""
Holds the DRAGONS version to be propagated throught all the DRAGONS package
and to be used in the documentation.
"""

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
        _version = "{:d}.{:d}".format(api, feature)

    else:
        _tag = '-{:s}'.format(tag) if len(tag) > 0 else ''
        _version = "{:d}.{:d}.{:d}".format(api, feature, bug) + _tag

    return _version
