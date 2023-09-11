import pytest
import astrodata
import astrodata.testing
import gemini_instruments
import os

THIS_DIR = os.path.dirname(__file__)

from lut_descriptors import fixture_data as descriptors_fixture_data

class FixtureIterator:
    def __init__(self, data_dict):
        self._data = data_dict

    def __iter__(self):
        for (instr, filename) in sorted(self._data.keys()):
            # ad = astrodata.open(os.path.join(THIS_DIR, 'test_data/{}/{}').format(instr, filename))
            ad = astrodata.open(astrodata.testing.download_from_archive(filename))
            for desc, value in self._data[(instr, filename)]:
                yield filename, ad, desc, value

@pytest.mark.parametrize("fn,ad,descriptor,value",
                         FixtureIterator(descriptors_fixture_data))
def test_descriptor(fn, ad, descriptor, value):
    method = getattr(ad, descriptor)
    if value is None:
        assert method() == None
    else:
        mvalue = method()
        if float in (type(value), type(mvalue)):
            assert abs(mvalue - value) < 0.0001
        else:
            assert value == mvalue
