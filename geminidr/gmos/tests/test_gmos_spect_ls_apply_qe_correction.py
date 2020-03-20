#!/usr/bin/env python

import glob
import os
import pandas as pd
import pytest
import shutil
import urllib
import xml.etree.ElementTree as et

import astrodata
import gemini_instruments

from astropy.utils.data import download_file
from contextlib import contextmanager
from geminidr.gmos import primitives_gmos_longslit
from gempy.adlibrary import dataselect
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals


URL = 'https://archive.gemini.edu/file/'

datasets = [
    ("N20180721S0444.fits")  # B1200 at 0.44 um
]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_locally_continuous(processed_ad):
    print(processed_ad)


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_globally_continuous():
    pass


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_stable():
    pass


# -- Fixtures -----------------------------------------------------------------
@pytest.fixture(scope='module')
def cache_path(new_path_to_inputs):
    """
    Factory as a fixture used to cache data and return its local path.

    Parameters
    ----------
    new_path_to_inputs : pytest.fixture
        Full path to cached folder.

    Returns
    -------
    function : Function used that downloads data from the archive and stores it
        locally.
    """
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
        local_path = os.path.join(new_path_to_inputs, filename)

        if not os.path.exists(local_path):
            tmp_path = download_file(URL + filename, cache=False)
            shutil.move(tmp_path, local_path)

            # `download_file` ignores Access Control List - fixing it
            os.chmod(local_path, 0o664)

        return local_path

    return _cache_path


def get_associated_calibrations(data_label):
    """
    Queries Gemini Observatory Archive for associated calibrations to reduce the
    data that will be used for testing.

    Parameters
    ----------
    data_label : str
        Input file datalabel.
    """
    url = "https://archive.gemini.edu/calmgr/{}".format(data_label)

    tree = et.parse(urllib.request.urlopen(url))
    root = tree.getroot()
    prefix = root.tag[:root.tag.rfind('}') + 1]

    def iter_nodes(node):
        cal_type = node.find(prefix + 'caltype').text
        filename = node.find(prefix + 'filename').text
        return filename, cal_type 

    cals = pd.DataFrame(
        [iter_nodes(node) for node in tree.iter(prefix + 'calibration')],
        columns=['filename', 'caltype'])

    cals = cals[~cals.caltype.str.contains('processed_')]
    cals = cals[~cals.caltype.str.contains('specphot')]
    cals = cals.drop(cals[cals.caltype.str.contains('bias')][5:].index)

    return cals.filename.values.tolist()


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
    contextmanager : A context manager function that allows easily changing
    folders.
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


@pytest.fixture(scope='module', params=datasets)
def processed_ad(
        request, cache_path, reduce_arc, reduce_bias, reduce_data, reduce_flat):

    filename = cache_path(request.param)
    ad = astrodata.open(filename)
    cals = [cache_path(c) for c in get_associated_calibrations(ad.data_label())]

    master_bias = reduce_bias(
        ad.data_label(), dataselect.select_data(cals, tags=['BIAS']))

    master_flat = reduce_flat(
        ad.data_label(), dataselect.select_data(cals, tags=['FLAT']), master_bias)

    master_arc = reduce_arc(
        ad.data_label(), dataselect.select_data(cals, tags=['ARC']))

    return reduce_data(ad, master_arc, master_bias, master_flat)


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
def reduce_data(output_path):
    """
    Factory for function for FLAT data reduction.

    Parameters
    ----------
    output_path : pytest.fixture
        Context manager used to write reduced data to a temporary folder.

    Returns
    -------
    function : A function that will read the standard star file, process them
    using a custom recipe and return an AstroData object.
    """
    def _reduce_data(ad, master_arc, master_bias, master_flat):
        with output_path():
            # Use config to prevent outputs when running Reduce via API
            logutils.config(file_name='log_{}.txt'.format(ad.data_label()))

            p = primitives_gmos_longslit.GMOSLongslit([ad])
            p.prepare()
            p.addDQ(static_bpm=None)
            p.addVAR(read_noise=True)
            p.overscanCorrect()
            p.biasCorrect(bias=master_bias)
            p.ADUToElectrons()
            p.addVAR(poisson_noise=True)
            p.flatCorrect(flat=master_flat)
            p.applyQECorrection(arc=master_arc)
            p.distortionCorrect(arc=master_arc)
            p.findSourceApertures(max_apertures=1)
            p.skyCorrectFromSlit()
            p.traceApertures()
            p.extract1DSpectra()
            p.linearizeSpectra()
            processed_ad = p.writeOutputs()

        return processed_ad
    return _reduce_data


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
    def _reduce_flat(datalabel, flat_fnames, master_bias):
        with output_path():
            logutils.config(file_name='log_flat_{}.txt'.format(datalabel))

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


if __name__ == '__main__':
    pytest.main()
