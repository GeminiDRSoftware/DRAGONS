"""
Fixtures to be used in tests in DRAGONS
"""

import os
import shutil
import urllib
import xml.etree.ElementTree as et
from contextlib import contextmanager

import pytest
from astropy.utils.data import download_file

URL = 'https://archive.gemini.edu/file/'


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


@pytest.fixture(scope='module')
def change_working_dir(path_to_outputs):
    """
    Factory that returns the output path as a context manager object, allowing
    easy access to the path to where the processed data should be stored.

    Parameters
    ----------
    path_to_outputs : pytest.fixture
        Fixture containing the root path to the output files.

    Returns
    -------
    contextmanager
        Enable easy change to temporary folder when reducing data.
    """
    path = os.path.join(path_to_outputs, "outputs")
    os.makedirs(path, exist_ok=True)

    @contextmanager
    def _change_working_dir():
        oldpwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(oldpwd)

    return _change_working_dir


def download_from_archive(filename, path='raw_files', env_var='DRAGONS_TEST'):
    """Download file from the archive and store it in the local cache.

    Parameters
    ----------
    filename : str
        The filename, e.g. N20160524S0119.fits
    path : str
        By default the file is stored at the root of the cache directory, but
        using ``path`` allows to specify a sub-directory.
    env_var: str
        Environment variable containing the path to the cache directory.

    Returns
    -------
    str
        Name of the cached file with the path added to it.
    """
    # Find cache path and make sure it exists
    cache_path = os.getenv(env_var)

    if cache_path is None:
        raise ValueError('Environment variable not set: {:s}'.format(env_var))
    elif not os.access(cache_path, os.W_OK):
        raise OSError('Could not access the path stored inside ${:s}. Make '
                      'sure the following path exists and that you have write '
                      'permissions in it: {:s}'.format(env_var, cache_path))

    cache_path = os.path.expanduser(cache_path)

    if path is not None:
        cache_path = os.path.join(cache_path, path)

    os.makedirs(cache_path, exist_ok=True)

    # Now check if the local file exists and download if not
    local_path = os.path.join(cache_path, filename)
    if not os.path.exists(local_path):
        tmp_path = download_file(URL + filename, cache=False)
        shutil.move(tmp_path, local_path)

        # `download_file` ignores Access Control List - fixing it
        os.chmod(local_path, 0o664)

    return local_path


@pytest.fixture(scope='module')
def path_to_inputs(request, path_to_test_data):
    """
    PyTest fixture that returns the path to where the input files for a given
    test module live.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.

    path_to_test_data : pytest.fixture
        Custom astrodata fixture that returs the root path to where input and
        reference files should live.

    Returns
    -------
    str:
        Path to the input files.
    """
    module_path = request.module.__name__.split('.') + ["inputs"]
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path_to_test_data, *module_path)

    if not os.path.exists(path):
        print(" Creating new directory to store input data for DRAGONS tests:"
              "\n    {:s}".format(path))
        os.makedirs(path)

    if not os.access(path, os.W_OK):
        pytest.fail('\n  Path to input test data exists but is not accessible: '
                    '\n    {:s}'.format(path))

    return path


@pytest.fixture(scope='module')
def path_to_refs(request, path_to_test_data):
    """
    PyTest fixture that returns the path to where the reference files for a
    given test module live.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.

    path_to_test_data : pytest.fixture
        Custom astrodata fixture that returs the root path to where input and
        reference files should live.

    Returns
    -------
    str:
        Path to the reference files.
    """
    module_path = request.module.__name__.split('.') + ["refs"]
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path_to_test_data, *module_path)

    if not os.path.exists(path):
        pytest.fail('\n Path to reference test data does not exist: '
                    '\n   {:s}'.format(path))

    if not os.access(path, os.W_OK):
        pytest.fail('\n Path to reference test data exists but is not accessible: '
                    '\n    {:s}'.format(path))

    return path


@pytest.fixture(scope='session')
def path_to_test_data(env_var='DRAGONS_TEST'):
    """
    PyTest fixture that reads the environment variable $DRAGONS_TEST that
    should contain data that will be used inside tests.

    If the environment variable does not exist, it marks the test to be skipped.

    If the environment variable exists but it not accessible, the test fails.

    Returns
    -------
    str : path to the reference data
    """
    path = os.getenv(env_var)

    if path is None:
        pytest.skip('Environment variable not set: $DRAGONS_TEST')

    path = os.path.expanduser(path).strip()

    if not os.access(path, os.W_OK):
        pytest.fail(
            '\n  Could not access the path stored inside $DRAGONS_TEST. '
            '\n  Make sure the following path exists and that you have '
            'write permissions in it:\n    {}'.format(path))

    return path


@pytest.fixture(scope='module')
def path_to_outputs(request, tmp_path_factory):
    """
    PyTest fixture that creates a temporary folder to save tests outputs.

    This output folder can be override via $DRAGONS_TEST_OUTPUTS environment
    variable or via `--basetemp` argument.

    Returns
    -------
    str
        Path to the output data.

    Raises
    ------
    IOError
        If output path does not exits.
    """
    if os.getenv('DRAGONS_TEST_OUTPUTS'):
        path = os.path.expanduser(os.getenv('DRAGONS_TEST_OUTPUTS'))
        if not os.path.exists(path):
            raise OSError(
                "Could not access path stored in $DRAGONS_TEST_OUTPUTS: "
                "{}\n Using current working directory".format(path))
    else:
        # path = str(tmp_path_factory.mktemp('dragons_tests', numbered=False))
        path = str(tmp_path_factory.getbasetemp())

    module_path = request.module.__name__.split('.')
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path, *module_path)
    os.makedirs(path, exist_ok=True)

    return path
