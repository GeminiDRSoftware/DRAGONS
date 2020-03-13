#!/usr/bin/env python
import os
import shutil
import urllib
import xml.etree.ElementTree as et
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from astropy.utils.data import download_file

import astrodata
from gempy.utils import logutils
from recipe_system.reduction.coreReduce import Reduce
from recipe_system.utils.reduce_utils import normalize_ucals

URL = 'https://archive.gemini.edu/file/'

datasets = [
    ("S20180707S0043.fits", "S20180707S0043_flat.fits"), # B600 @ 0.520
]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("fname, ref_fname", datasets)
def test_processed_flat_has_median_around_one(fname, ref_fname, processed_flat):

    ad = processed_flat(fname)
    np.testing.assert_almost_equal(np.median(ad[0].data.ravel()), 1.0, decimal=3)


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("fname, ref_fname", datasets)
def test_processed_flat_is_stable(fname, ref_fname, processed_flat, reference_flat):

    ad = processed_flat(fname)
    ad_ref = reference_flat(ref_fname)

    for ext, ext_ref in zip(ad, ad_ref):
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
    def _processed_flat(fname):
        flat_fname = cache_path(fname)
        data_label = query_datalabel(flat_fname)

        bias_fnames = query_associated_bias(data_label)
        bias_fnames = [cache_path(fname) for fname in bias_fnames]

        master_bias = reduce_bias(data_label, bias_fnames)
        flat_ad = reduce_flat(data_label, flat_fname, master_bias)

        return flat_ad
    return _processed_flat


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
    Query datalabel associated to the filename from the Gemini Archive.

    Parameters
    ----------
    fname : str
        Input file name.

    Returns
    -------
    str : Data label.
    """
    json_summmary_url = 'https://archive.gemini.edu/jsonsummary/{:s}'
    df = pd.read_json(json_summmary_url.format(fname))
    return df.iloc[0]['data_label']


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
            reduce.ucals = normalize_ucals(reduce.files, calibration_files)
            reduce.runr()

            master_flat = reduce.output_filenames.pop()
            master_flat_ad = astrodata.open(master_flat)

        return master_flat_ad

    return _reduce_flat


@pytest.fixture
def reference_flat(new_path_to_refs):
    def _reference_flat(ref_fname):

        ref_path = os.path.join(new_path_to_refs, ref_fname)

        if not os.path.exists(ref_path):
            pytest.fail('\n  Referece file does not exists: '
                        '\n    {:s}'.format(ref_path))

        return astrodata.open(ref_path)
    return _reference_flat


if __name__ == '__main__':
    pytest.main()
