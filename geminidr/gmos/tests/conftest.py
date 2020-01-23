#!/usr/bin/env python
"""
Configuration file for tests in `geminidr.gmos.tests`
"""
import os

import pytest

import astrodata
from astrodata import testing
from gempy.utils import logutils


@pytest.fixture(scope="session", autouse=True)
def setup_log(path_to_outputs):
    """
    Fixture that setups DRAGONS' logging system to avoid duplicated outputs.

    Parameters
    ----------
    path_to_outputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to
        the output folder.
    """
    log_file = "{}.log".format(os.path.splitext(os.path.basename(__file__))[0])
    log_file = os.path.join(path_to_outputs, log_file)

    logutils.config(mode="standard", file_name=log_file)


@pytest.fixture(scope='module')
def ad_ref(request, path_to_refs):
    """
    Loads existing reference FITS files as AstroData objects.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    path_to_refs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached reference files.

    Returns
    -------
    AstroData
        Object containing Wavelength Solution table.

    Raises
    ------
    IOError
        If the reference file does not exist. It should be created and verified
        manually.
    """
    fname = os.path.join(path_to_refs, request.param)

    if not os.path.exists(fname):
        raise IOError(" Cannot find reference file:\n {:s}".format(fname))

    return astrodata.open(fname)


@pytest.fixture(scope="module")
def ad_factory(request, path_to_inputs):
    """
    Custom fixture that loads existing cached input data. If the input file
    does not exists and PyTest is called with `--force-preprocess-data`, it
    downloads and cache the raw data and preprocess it using `recipe` and its
    arguments.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.
    path_to_inputs : fixture
        Custom fixture defined in `astrodata.testing` containing the path to the
        cached input files.

    Returns
    -------
    function
        Callable responsable to download/preprocess/cache/load the input data.

    Raises
    ------
    IOError
        If the input file does not exist and if --force-preprocess-data is False.

    """
    force_preprocess = request.config.getoption("--force-preprocess-data")

    def _ad_factory(filename, recipe, **kwargs):

        filepath = os.path.join(path_to_inputs, filename)

        if os.path.exists(filepath):
            print("Loading existing input file:  {}".format(filename))
            _ad = astrodata.open(filepath)

        elif force_preprocess:
            print("Pre-processing input file:  {}".format(filename))
            subpath, basename = os.path.split(filepath)
            basename, extension = os.path.splitext(basename)
            basename = basename.split('_')[0] + extension

            raw_fname = testing.download_from_archive(basename, path=subpath)

            _ad = astrodata.open(raw_fname)
            _ad = recipe(_ad, os.path.join(path_to_inputs, subpath), **kwargs)

        else:
            raise IOError(
                "Cannot find input file:\n {:s}\n".format(filepath) +
                "Run PyTest with --force-preprocessed-data if you want to "
                "force data cache and preprocessing.")

        return _ad

    return _ad_factory
