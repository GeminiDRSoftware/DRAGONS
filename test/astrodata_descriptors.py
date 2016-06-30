import pytest
import astrodata
import instruments

from lut_descriptors import fixture_data as descriptors_fixture_data

class FixtureIterator(object):
    def __init__(self, data_dict):
        self._data = data_dict

    def __iter__(self):
        for (instr, filename) in sorted(self._data.keys()):
            ad = astrodata.open('data/{}/{}'.format(instr, filename))
            for desc, value in self._data[(instr, filename)]:
                yield ad, desc, value

@pytest.mark.parametrize("ad,descriptor,value", FixtureIterator(descriptors_fixture_data))
def test_descriptor(ad,descriptor,value):
    method = getattr(ad, descriptor)
    if value is None:
        with pytest.raises(Exception):
            method()
    else:
        assert method() == value
