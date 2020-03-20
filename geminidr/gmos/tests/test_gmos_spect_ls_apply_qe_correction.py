#!/usr/bin/env python

import pandas as pd
import pytest
import xml.etree.ElementTree as et

from urllib import request


datasets = [
    # (datalabel, )
    ("GN-2018B-Q-313-5-001")  # B1200 at 0.44 um
]

# -- Tests --------------------------------------------------------------------
@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("processed_ad", datasets, indirect=True)
def test_applied_qe_is_locally_continuous(processed_ad):
    pass


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_globally_continuous():
    pass


@pytest.mark.gmosls
@pytest.mark.dragons_remote_data
def test_applied_qe_is_stable():
    pass


# -- Fixtures -----------------------------------------------------------------
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

    tree = et.parse(request.urlopen(url))
    root = tree.getroot()
    prefix = root.tag[:root.tag.rfind('}') + 1]

    def iter_nodes(node):
        cal_type = node.find(prefix + 'caltype').text
        filename = node.find(prefix + 'filename').text
        return cal_type, filename

    cals = pd.DataFrame(
        [iter_nodes(node) for node in tree.iter(prefix + 'calibration')],
        columns=['caltype', 'filename'])

    cals = cals[~cals.caltype.str.contains('processed_')]
    cals = cals[~cals.caltype.str.contains('specphot')]
    cals = cals.drop(cals[cals.caltype.str.contains('bias')][5:].index)

    return cals.values.tolist()


@pytest.fixture(scope='module')
def processed_ad(request):
    data_label = request.param
    calibrations = get_associated_calibrations(data_label)
    print(calibrations)


if __name__ == '__main__':
    pytest.main()
