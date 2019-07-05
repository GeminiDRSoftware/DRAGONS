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


def get_git_hash():
    """
    Returns the current git hash to be used in the versioning system.

    Returns
    -------
    str : git hash
    """
    from subprocess import check_output

    git_hash = str(check_output(["git", "describe", "--always"]).strip())
    git_hash = git_hash.split("-")[-1]

    return git_hash


def version(short=False, tag=TAG):
    """
    Returns DRAGONS's version based on the api,
    feature and bug numbers.

    Returns
    -------
    str : formatted version
    """
    tag = get_git_hash() if not tag else tag

    if short:
        _version = "{:d}.{:d}".format(API, FEATURE)

    else:
        _tag = '_{:s}'.format(tag)
        _version = "{:d}.{:d}.{:d}".format(API, FEATURE, BUG) + _tag

    return _version
