import os
import pytest
import glob
import warnings

import astrodata
import gemini_instruments
from .conftest import test_path


try:
    path = os.environ['TEST_PATH']
except KeyError:
    warnings.warn("Could not find environment variable: $TEST_PATH")
    path = ''

if not os.path.exists(path):
    warnings.warn("Could not find path stored in $TEST_PATH: {}".format(path))
    path = ''

archivefiles = glob.glob(os.path.join(path, "Archive/", "*fits"))





# Opens all files in archive
@pytest.mark.parametrize('filename', archivefiles)
def test_all_descriptors_can_parse(filename):
    ad = astrodata.open(filename)
    typelist = []
    for descriptor in ad.descriptors:
        try:
            typelist.append((descriptor, type(getattr(ad, descriptor)())))
        except Exception as err:
            #print("{} failed on call: {}".format(descriptor, str(err)))
            warnings.warn("{} failed in the filename {},"
                          " on call: {}".format(
                descriptor, filename, str(err)))
