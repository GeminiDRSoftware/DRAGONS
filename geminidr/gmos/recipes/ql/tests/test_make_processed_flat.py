#!/usr/bin/env python
import os
import urllib
import xml.etree.ElementTree as et

import pandas as pd
import pytest

from astrodata import testing

datasets = [
    "S20180707S0043.fits",  # B600 @ 0.520
]


@pytest.fixture
def processed_flat(request, path_to_inputs):
    raw_filename = request.param
    _ = testing.download_from_archive(raw_filename, path=path_to_inputs)

    data_label = query_datalabel(raw_filename)
    list_of_bias = query_associated_bias(data_label)
    list_of_bias = [testing.download_from_archive(bias_fname, path=path_to_inputs)
                    for bias_fname in list_of_bias]
    _ = [print(bias_fname) for bias_fname in list_of_bias]

    suffix = "_flat"

    processed_filename, ext = os.path.splitext(raw_filename)
    processed_filename += "_{:s}{:s}".format(suffix, ext)

    return


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


@pytest.fixture
def reference_flat():
    return


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("processed_flat", datasets, indirect=True)
def test_processed_flat_has_median_around_one(processed_flat):
    assert False


@pytest.mark.gmosls
@pytest.mark.parametrize("processed_flat, reference_flat", zip(datasets, datasets))
def test_processed_flat_is_stable(processed_flat, reference_flat):
    pass


if __name__ == '__main__':
    pytest.main()
