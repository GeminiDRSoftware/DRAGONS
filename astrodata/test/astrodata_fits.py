import pytest
import astrodata
import gemini_instruments
import os

THIS_DIR = os.path.dirname(__file__)

from lut_descriptors import fixture_data as descriptors_fixture_data

# Tests to perform:

# Opening a FITS file
# Length
# Slicing
# Slicing to single


class FixtureIterator(object):
    def __init__(self, data_dict):
        self._data = data_dict

    def __iter__(self):
        for (instr, filename) in sorted(self._data.keys()):
            ad = astrodata.open(os.path.join(THIS_DIR, 'test_data/{}/{}').format(instr, filename))
            for desc, value in self._data[(instr, filename)]:
                yield filename, ad, desc, value
