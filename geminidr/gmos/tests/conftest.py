#!/usr/bin/env python
"""
Configuration file for tests in `geminidr.gmos.tests`
"""
import os
import pytest
import shutil

from astropy.utils.data import download_file
from contextlib import contextmanager

import astrodata
from astrodata import testing
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

URL = 'https://archive.gemini.edu/file/'


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
        raise OSError(" Cannot find reference file:\n {:s}".format(fname))

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
        Custom fixture defined in `astrodata.testing` containing the path
        to the cached input files.

    Returns
    -------
    function
        Callable responsable to download/preprocess/cache/load the input data.

    Raises
    ------
    IOError
        If the input file does not exist and if ``--force-preprocess-data``
        is False.

    """
    force_preprocess = request.config.getoption("--force-preprocess-data")

    def _ad_factory(filename, recipe, **kwargs):

        filepath = os.path.join(path_to_inputs, filename)

        if os.path.exists(filepath):
            print("Loading existing input file: {}".format(filename))
            _ad = astrodata.open(filepath)

        elif force_preprocess:
            print("Pre-processing input file: {}".format(filename))
            subpath, basename = os.path.split(filepath)
            basename, extension = os.path.splitext(basename)
            basename = basename.split('_')[0] + extension

            raw_fname = testing.download_from_archive(basename, path=subpath)

            _ad = astrodata.open(raw_fname)
            _ad = recipe(_ad, os.path.join(path_to_inputs, subpath), **kwargs)

        else:
            raise OSError(
                "Cannot find input file:\n {:s}\n".format(filepath) +
                "Run PyTest with --force-preprocessed-data if you want to "
                "force data cache and preprocessing.")

        return _ad

    return _ad_factory


@pytest.fixture(scope='module')
def cache_path(request, path_to_outputs):
    """
    Factory as a fixture used to cache data and return its local path.

    Parameters
    ----------
    request : pytest.fixture
        Fixture that contains information this fixture's parent.
    path_to_outputs : pytest.fixture
        Full path to root cache folder.

    Returns
    -------
    function : Function used that downloads data from the archive and stores it
        locally.
    """
    module_path = request.module.__name__.split('.')
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path_to_outputs, *module_path)
    os.makedirs(path, exist_ok=True)

    def _cache_path(filename):
        """
        Download data from Gemini Observatory Archive and cache it locally.

        Parameters
        ----------
        filename : str
            The filename, e.g. N20160524S0119.fits

        Returns
        -------
        str : full path of the cached file.
        """
        local_path = os.path.join(path, filename)

        if not os.path.exists(local_path):
            tmp_path = download_file(URL + filename, cache=False)
            shutil.move(tmp_path, local_path)

            # `download_file` ignores Access Control List - fixing it
            os.chmod(local_path, 0o664)

        return local_path

    return _cache_path


@pytest.fixture(scope='module')
def get_master_arc(new_path_to_inputs, output_path):
    """
    Factory that creates a function that reads the master arc file from the
    permanent input folder or from the temporarly local cache, depending on
    command line options.

    Parameters
    ----------
    new_path_to_inputs : pytest.fixture
        Path to the permanent local input files.
    output_path : contextmanager
        Enable easy change to temporary folder when reducing data.

    Returns
    -------
    AstroData
        The master arc.
    """

    def _get_master_arc(ad, pre_process):

        cals = testing.get_associated_calibrations(
            ad.filename.split('_')[0] + '.fits')

        arc_filename = cals[cals.caltype == 'arc'].filename.values[0]
        arc_filename = arc_filename.split('.fits')[0] + '_arc.fits'

        if pre_process:
            with output_path():
                master_arc = astrodata.open(arc_filename)
        else:
            master_arc = astrodata.open(
                os.path.join(new_path_to_inputs, arc_filename))

        return master_arc

    return _get_master_arc


@pytest.fixture(scope='module')
def output_path(request, path_to_outputs):
    """
    Factory that returns the output path as a context manager object, allowing
    easy access to the path to where the processed data should be stored.

    Parameters
    ----------
    request : pytest.fixture
        Fixture that contains information this fixture's parent.
    path_to_outputs : pytest.fixture
        Fixture containing the root path to the output files.

    Returns
    -------
    contextmanager
        Enable easy change to temporary folder when reducing data.
    """
    module_path = request.module.__name__.split('.') + ["outputs"]
    module_path = [item for item in module_path if item not in "tests"]
    path = os.path.join(path_to_outputs, *module_path)

    os.makedirs(path, exist_ok=True)

    @contextmanager
    def _output_path():
        oldpwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(oldpwd)

    return _output_path


@pytest.fixture(scope='module')
def reduce_arc(output_path):
    """
    Factory for function for ARCS data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the arcs files, process them and
    return the name of the master arc.
    """

    def _reduce_arc(dlabel, arc_fnames):
        with output_path():
            # Use config to prevent duplicated outputs when running Reduce via API
            logutils.config(file_name='log_arc_{}.txt'.format(dlabel))

            reduce = Reduce()
            reduce.files.extend(arc_fnames)
            reduce.runr()

            master_arc = reduce.output_filenames.pop()
        return master_arc

    return _reduce_arc


@pytest.fixture(scope='module')
def reduce_bias(output_path):
    """
    Factory for function for BIAS data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the bias files, process them and
    return the name of the master bias.
    """

    def _reduce_bias(datalabel, bias_fnames):
        with output_path():
            logutils.config(file_name='log_bias_{}.txt'.format(datalabel))

            reduce = Reduce()
            reduce.files.extend(bias_fnames)
            reduce.runr()

            master_bias = reduce.output_filenames.pop()

        return master_bias

    return _reduce_bias


@pytest.fixture(scope='module')
def reduce_flat(output_path):
    """
    Factory for function for FLAT data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the flat files, process them and
    return the name of the master flat.
    """
    def _reduce_flat(data_label, flat_fnames, master_bias):
        with output_path():
            logutils.config(file_name='log_flat_{}.txt'.format(data_label))

            calibration_files = ['processed_bias:{}'.format(master_bias)]

            reduce = Reduce()
            reduce.files.extend(flat_fnames)
            reduce.mode = 'ql'
            reduce.ucals = normalize_ucals(reduce.files, calibration_files)
            reduce.runr()

            master_flat = reduce.output_filenames.pop()
            master_flat_ad = astrodata.open(master_flat)

        return master_flat_ad
    return _reduce_flat


@pytest.fixture(scope="module")
def reference_ad(new_path_to_refs):
    """
    Read the reference file.

    Parameters
    ----------
    new_path_to_refs : pytest.fixture
        Fixture containing the root path to the reference files.

    Returns
    -------
    function : function that loads the reference file.
    """
    def _reference_ad(filename):
        path = os.path.join(new_path_to_refs, filename)
        return astrodata.open(path)
    return _reference_ad


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