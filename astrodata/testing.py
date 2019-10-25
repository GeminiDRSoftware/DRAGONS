#!/usr/bin/env python
"""
Fixtures to be used in tests in DRAGONS
"""

import os
import pytest
import shutil
import warnings
from astropy.utils.data import download_file

URL = 'https://archive.gemini.edu/file/'
DEFAULT_CACHE_DIRECTORY = '~/.geminidr/cache'


def download_from_archive(filename, path=None):
    """Download file from the archive and store it in the local cache.

    Parameters
    ----------
    filename : str
        The filename, e.g. N20160524S0119.fits
    path : str
        By default the file is stored at the root of the cache directory, but
        using ``path`` allows to specify a sub-directory.

    """
    # Find cache path and make sure it exists
    cache_path = os.getenv('DRAGONS_TEST_INPUTS', DEFAULT_CACHE_DIRECTORY)
    cache_path = os.path.expanduser(cache_path)
    if path is not None:
        cache_path = os.path.join(cache_path, path)
    os.makedirs(cache_path, exist_ok=True)

    # Now check if the local file exists and download if not
    local_path = os.path.join(cache_path, filename)
    if not os.path.exists(local_path):
        tmp_path = download_file(URL + filename, cache=False)
        shutil.move(tmp_path, local_path)

    return local_path


def assert_have_same_distortion(ad, ad_ref):
    """
    Checks if two :class:`~astrodata.AstroData` (or any subclass) have the
    same distortion.

    Parameters
    ----------
        ad : :class:`astrodata.AstroData`
            AstroData object to be checked.
        ad_ref : :class:`astrodata.AstroData`
            AstroData object used as reference
    """
    from gempy.library.astromodels import dict_to_chebyshev
    from numpy.testing import assert_allclose

    for ext, ext_ref in zip(ad, ad_ref):

        distortion = dict(zip(ext.FITCOORD["name"], ext.FITCOORD["coefficients"]))
        distortion = dict_to_chebyshev(distortion)

        distortion_ref = dict(zip(ext.FITCOORD["name"], ext.FITCOORD["coefficients"]))
        distortion_ref = dict_to_chebyshev(distortion_ref)

        assert isinstance(distortion, type(distortion_ref))
        assert_allclose(distortion.parameters, distortion_ref.parameters)


def assert_same_class(ad, ad_ref):
    """
    Compare if two :class:`~astrodata.AstroData` (or any subclass) have the
    same class.

    Parameters
    ----------
        ad : :class:`astrodata.AstroData` or any subclass
            AstroData object to be checked.
        ad_ref : :class:`astrodata.AstroData` or any subclass
            AstroData object used as reference
    """
    from astrodata import AstroData

    assert isinstance(ad, AstroData)
    assert isinstance(ad_ref, AstroData)
    assert isinstance(ad, type(ad_ref))


def assert_wavelength_solutions_are_close(ad, ad_ref):
    """
    Checks if two :class:`~astrodata.AstroData` (or any subclass) have the
    wavelength solution.

    Parameters
    ----------
        ad : :class:`astrodata.AstroData` or any subclass
            AstroData object to be checked.
        ad_ref : :class:`astrodata.AstroData` or any subclass
            AstroData object used as reference
    """
    from gempy.library.astromodels import dict_to_chebyshev
    from numpy.testing import assert_allclose

    for ext, ext_ref in zip(ad, ad_ref):

        assert hasattr(ext, "WAVECAL")
        wcal = dict(zip(ext.WAVECAL["name"], ext.WAVECAL["coefficients"]))
        wcal = dict_to_chebyshev(wcal)

        assert hasattr(ext_ref, "WAVECAL")
        wcal_ref = dict(zip(ad[0].WAVECAL["name"], ad[0].WAVECAL["coefficients"]))
        wcal_ref = dict_to_chebyshev(wcal_ref)

        assert isinstance(wcal, type(wcal_ref))
        assert_allclose(wcal.parameters, wcal_ref.parameters)


@pytest.fixture(scope='module')
def path_to_inputs():
    """
    PyTest fixture that reads the environment variable $DRAGONS_TEST_INPUTS that
    should contains input data for testing.

    If the environment variable does not exist, it marks the test to be skipped.

    If the environment variable exists but it not accessible, it also marks the
    test to be skipped.

    The skip reason changes depending on which situation causes it to be skipped.

    Returns
    -------
        str : path to the input data
    """

    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_INPUTS'])
        path = path.strip()
    except KeyError:
        pytest.skip(
            "Could not find environment variable: $DRAGONS_TEST_INPUTS")

    # noinspection PyUnboundLocalVariable
    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_INPUTS: "
            "{}".format(path)
        )

    return path


@pytest.fixture(scope='module')
def path_to_refs():
    """
    PyTest fixture that reads the environment variable $DRAGONS_TEST_REFS that
    should contains reference data for testing.

    If the environment variable does not exist, it marks the test to be skipped.

    If the environment variable exists but it not accessible, it also marks the
    test to be skipped.

    The skip reason changes depending on which situation causes it to be skipped.

    Returns
    -------
        str : path to the reference data
    """
    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_REFS'])
    except KeyError:
        pytest.skip(
            "Could not find environment variable: $DRAGONS_TEST_REFS")

    # noinspection PyUnboundLocalVariable
    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_REFS: "
            "{}".format(path)
        )

    return path


@pytest.fixture(scope='module')
def path_to_outputs():
    """
    PyTest fixture that reads the environment variable $DRAGONS_TEST_OUTPUTS
    where output data will be stored during the tests.

    If the environment variable does not exist, it marks the test to be skipped.

    If the environment variable exists but it not accessible, it also marks the
    test to be skipped.

    The skip reason changes depending on which situation causes it to be skipped.

    Returns
    -------
        str : path to the output data
    """
    try:
        path = os.path.expanduser(os.environ['DRAGONS_TEST_OUTPUTS'])
    except KeyError:
        warnings.warn("Could not find environment variable: $DRAGONS_TEST_OUTPUTS"
                      "\n Using current working directory")
        path = os.getcwd()

    if not os.path.exists(path):
        warnings.warn(
            "Could not access path stored in $DRAGONS_TEST_OUTPUTS: "
            "{}".format(path) +
            "\n Using current working directory")
        path = os.getcwd()

    return path
