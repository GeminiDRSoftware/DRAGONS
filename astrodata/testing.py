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


def download_from_archive(filename, sub_path='raw_files', env_var='DRAGONS_TEST'):
    """Download file from the archive and store it in the local cache.

    Parameters
    ----------
    filename : str
        The filename, e.g. N20160524S0119.fits
    sub_path : str
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
    root_cache_path = os.getenv(env_var)

    if root_cache_path is None:
        raise ValueError('Environment variable not set: {:s}'.format(env_var))

    root_cache_path = os.path.expanduser(root_cache_path)

    if sub_path is not None:
        cache_path = os.path.join(root_cache_path, sub_path)

    if os.path.exists(cache_path):
        print('Folder exists. Skipping creation.\n {:s}'.format(cache_path))
    else:
        print('Create new folder:\n {:s}'.format(cache_path))
        os.makedirs(cache_path, exist_ok=True)

    # Now check if the local file exists and download if not
    local_path = os.path.join(cache_path, filename)
    if not os.path.exists(local_path):
        tmp_path = download_file(URL + filename, cache=False)
        shutil.move(tmp_path, local_path)

        # `download_file` ignores Access Control List - fixing it
        os.chmod(local_path, 0o664)

    return local_path


def get_associated_calibrations(filename, nbias=5):
    """
    Queries Gemini Observatory Archive for associated calibrations to reduce the
    data that will be used for testing.
    Parameters
    ----------
    filename : str
        Input file name
    """
    pd = pytest.importorskip("pandas", minversion='1.0.0')
    url = "https://archive.gemini.edu/calmgr/{}".format(filename)

    tree = et.parse(urllib.request.urlopen(url))
    root = tree.getroot()
    prefix = root.tag[:root.tag.rfind('}') + 1]

    def iter_nodes(node):
        cal_type = node.find(prefix + 'caltype').text
        cal_filename = node.find(prefix + 'filename').text
        return cal_filename, cal_type

    cals = pd.DataFrame(
        [iter_nodes(node) for node in tree.iter(prefix + 'calibration')],
        columns=['filename', 'caltype'])

    cals = cals.sort_values(by='filename')
    cals = cals[~cals.caltype.str.contains('processed_')]
    cals = cals[~cals.caltype.str.contains('specphot')]
    cals = cals.drop(cals[cals.caltype.str.contains('bias')][nbias:].index)

    return cals


@pytest.fixture(scope='module')
def path_to_inputs(request, env_var='DRAGONS_TEST'):
    """
    PyTest fixture that returns the path to where the input files for a given
    test module live.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.

    env_var : str
        Environment variable that contains the root path to the input data.

    Returns
    -------
    str:
        Path to the input files.
    """
    path_to_test_data = os.getenv(env_var)

    if path_to_test_data is None:
        pytest.skip('Environment variable not set: $DRAGONS_TEST')

    path_to_test_data = os.path.expanduser(path_to_test_data).strip()

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
def path_to_refs(request, env_var='DRAGONS_TEST'):
    """
    PyTest fixture that returns the path to where the reference files for a
    given test module live.

    Parameters
    ----------
    request : fixture
        PyTest's built-in fixture with information about the test itself.

    env_var : str
        Environment variable that contains the root path to the input data.

    Returns
    -------
    str:
        Path to the reference files.
    """
    path_to_test_data = os.getenv(env_var)

    if path_to_test_data is None:
        pytest.skip('Environment variable not set: $DRAGONS_TEST')

    path_to_test_data = os.path.expanduser(path_to_test_data).strip()

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
        path = str(tmp_path_factory.getbasetemp())

    module_path = request.module.__name__.split('.')
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path, *module_path)
    os.makedirs(path, exist_ok=True)

    return path
