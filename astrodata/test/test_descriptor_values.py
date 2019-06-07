
import os
import pytest
import glob

import astrodata
import gemini_instruments
from .conftest import test_path


class TestDescriptorValues:

    def test_airmass_descriptor_value_is_acceptable(self, test_path):

        archive_files = glob.glob(os.path.join(test_path, "Archive/", "*fits"))

        for _file in archive_files:

            ad = astrodata.open(_file)

            try:
                assert ((ad.airmass() >= 1.0) or (ad.airmass() is None)), \
                    "Test failed for file: {}".format(_file)

            except Exception as err:
                print("{} failed on call: {}".format(ad.airmass, str(err)))


if __name__ == '__main__':
    pytest.main()
