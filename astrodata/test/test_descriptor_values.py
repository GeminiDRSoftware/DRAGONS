
import os
import pytest
import glob
import warnings
import datetime

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


# Returns list of all files in the TEST_PATH directory
archive_files = glob.glob(os.path.join(path, "Archive/", "*fits"))

# Separates the directory from the list, helps cleanup code
fits_files = [_file.split('/')[-1] for _file in archive_files]
full_dir = archive_files[0].rsplit('/', 1)[0]

# Fixtures for module and class
@pytest.fixture(scope='class')
def setup_test_descriptor_values(request):
    print('setup TestDescriptorValues')

    def fin():
        print('\nteardown TestDescriptorValues')
    request.addfinalizer(fin)
    return


@pytest.mark.usefixtures('setup_test_descriptor_values')
class TestDescriptorValues:

    @pytest.mark.parametrize("filename", fits_files)
    def test_airmass_descriptor_value_is_acceptable(self, test_path, filename):
        ad = astrodata.open(os.path.join(full_dir, filename))
        try:
            assert ((ad.airmass() >= 1.0)
                    or (ad.airmass() is None))
        except Exception as err:
            print("{} failed on call: {}".format(ad.airmass, str(err)))


    # @pytest.mark.parametrize("filename", archivefiles)
    # def test_azimuth_descriptor_is_none_or_float(self, filename):
    #     ad = astrodata.open(filename)
    #     assert ((0.0 <= ad.azimuth() <= 360.0)
    #             or (ad.azimuth() is None))
    #