import pytest
import astrodata
import instruments

from lut_tags import fixture_data as tags_fixture_data

class FixtureIterator(object):
    def __init__(self, data_dict):
        self._data = data_dict

    def __iter__(self):
        for key in sorted(self._data.keys()):
            (instr, filename) = key
            ad = astrodata.open('data/{}/{}'.format(instr, filename))
            yield filename, ad, set(self._data[key])

@pytest.mark.parametrize("fn,ad,tag_set", FixtureIterator(tags_fixture_data))
def test_descriptor(fn, ad, tag_set):
    assert ad.tags == tag_set
