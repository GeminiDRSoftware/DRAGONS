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
    path = ''
if not os.path.exists(path):
    path = ''

# Returns list of all files in the TEST_PATH directory
archive_files = glob.glob(os.path.join(path, "Archive/", "*fits"))

# Separates the directory from the list, helps cleanup code
fits_files = [_file.split('/')[-1] for _file in archive_files]


# Opens all files in archive
@pytest.mark.parametrize('filename', fits_files)
def test_all_descriptors_can_parse(test_path, filename):
    ad = astrodata.open(os.path.join(test_path, "Archive/", filename))
    typelist = []
    for descriptor in ad.descriptors:
        try:
            typelist.append((descriptor, type(getattr(ad, descriptor)())))
        except Exception as err:
            #print("{} failed on call: {}".format(descriptor, str(err)))
            warnings.warn("{} failed in the filename {},"
                          " on call: {}".format(
                descriptor, filename, str(err)))
