#!/usr/bin/env python
import os
import shutil
import urllib
import xml.etree.ElementTree as et
from contextlib import contextmanager

import numpy as np
import pytest
from astropy.utils.data import download_file

import astrodata
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

URL = 'https://archive.gemini.edu/file/'

datasets = [
    "S20180707S0043.fits",  # B600 at 0.520 um
    "S20190502S0096.fits",  # B600 at 0.525 um
    "S20200122S0020.fits",  # B600 at 0.520 um
    "N20200101S0055.fits",  # B1200 at 0.495 um
    # "S20180410S0120.fits",  # B1200 at 0.595 um  # Scattered light?
    # "S20190410S0053.fits",  # B1200 at 0.463 um  # Scattered light?

]

refs = ["_flat".join(os.path.splitext(f)) for f in datasets]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
def test_processed_flat_has_median_around_one(processed_flat):
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_almost_equal(np.median(data.ravel()), 1.0, decimal=3)


# @pytest.mark.skip(reason="High std on half of datasets. Scattered light?")
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
def test_processed_flat_has_small_std(processed_flat):
    for ext in processed_flat:
        data = np.ma.masked_array(ext.data, mask=ext.mask)
        np.testing.assert_array_less(np.std(data.ravel()), 0.1)


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize(
    "processed_flat, processed_ref_flat", zip(datasets, refs), indirect=True)
def test_processed_flat_is_stable(processed_flat, processed_ref_flat):
    for ext, ext_ref in zip(processed_flat, processed_ref_flat):
        np.testing.assert_allclose(ext.data, ext_ref.data, rtol=1e-7)


# -- Fixtures ----------------------------------------------------------------
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


@pytest.fixture(scope='module')
def output_path(request, path_to_outputs):
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
def processed_flat(request, cache_path, reduce_bias, reduce_flat):
    flat_fname = cache_path(request.param)
    data_label = query_datalabel(flat_fname)

    bias_fnames = query_associated_bias(data_label)
    bias_fnames = [cache_path(fname) for fname in bias_fnames]

    master_bias = reduce_bias(data_label, bias_fnames)
    flat_ad = reduce_flat(data_label, flat_fname, master_bias)

    return flat_ad


@pytest.fixture
def processed_ref_flat(request, new_path_to_refs):
    filename = request.param
    ref_path = os.path.join(new_path_to_refs, filename)

    if not os.path.exists(ref_path):
        pytest.fail('\n  Reference file does not exists: '
                    '\n    {:s}'.format(ref_path))

    return astrodata.open(ref_path)


def query_associated_bias(data_label):
    """
    Queries Gemini Observatory Archive for associated bias calibration files
    for a batch reduction.

    Parameters
    ----------
    data_label : str
        Flat data label.

    Returns
    -------
    list : list with five bias files.
    """
    cal_url = "https://archive.gemini.edu/calmgr/bias/{}"
    tree = et.parse(urllib.request.urlopen(cal_url.format(data_label)))

    root = tree.getroot()[0]

    bias_files = []

    # root[0] = datalabel, root[1] = filename, root[2] = md5
    for e in root[3:]:
        [caltype, datalabel, filename, md5, url] = [ee.text for ee in e if 'calibration' in e.tag]
        bias_files.append(filename)

    # I am interested in only five bias per flat file
    return bias_files[:5] if len(bias_files) > 0 else bias_files


def query_datalabel(fname):
    """
    Retrieve the data label associated to the input file.

    Parameters
    ----------
    fname : str
        Input file name.

    Returns
    -------
    str : Data label.
    """
    ad = astrodata.open(fname)
    return ad.data_label()


@pytest.fixture(scope='module')
def reduce_bias(output_path):
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
    def _reduce_flat(datalabel, flat_fname, master_bias):
        with output_path():
            logutils.config(file_name='log_flat_{}.txt'.format(datalabel))

            calibration_files = ['processed_bias:{}'.format(master_bias)]

            reduce = Reduce()
            reduce.files.extend([flat_fname])
            reduce.mode = 'ql'
            reduce.ucals = normalize_ucals(reduce.files, calibration_files)
            reduce.runr()

            master_flat = reduce.output_filenames.pop()
            master_flat_ad = astrodata.open(master_flat)

        return master_flat_ad
    return _reduce_flat


if __name__ == '__main__':
    pytest.main()
