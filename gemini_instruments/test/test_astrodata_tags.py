import pytest
import astrodata
import astrodata.testing
import gemini_instruments
import os

from .lut_tags import fixture_data as tags_fixture_data

class FixtureIterator:
    def __init__(self, data_dict):
        self._data = data_dict

        # The `path_to_inputs` fixture returns $DRAGONS_TEST/test_astrodata_tags
        # here, which feels weird, so manually create the path to input data as
        # $DRAGONS_TEST/gemini_instruments/test_astrodata_tags/inputs
        # DB 20230907
        path_to_test_data = os.getenv("DRAGONS_TEST")
        if path_to_test_data is None:
            pytest.skip('Environment variable not set: $DRAGONS_TEST')
        path_to_test_data = os.path.expanduser(path_to_test_data).strip()
        path_to_files = ['gemini_instruments', 'test_astrodata_tags',
                         'inputs']
        self.path_to_test_data = os.path.join(path_to_test_data,
                                              *path_to_files)

    def __iter__(self):
        for key in sorted(self._data.keys()):
            (instr, filename) = key
            if '_' in filename:
                filepath = os.path.join(self.path_to_test_data, filename)
                # Allow this to run as a github workflow that won't have
                # access to these files
                try:
                    ad = astrodata.open(filepath)
                except FileNotFoundError:
                    ad = None
            else:
                ad = astrodata.open(
                    astrodata.testing.download_from_archive(filename))
            yield filename, ad, set(self._data[key])

@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("fn,ad,tag_set", FixtureIterator(tags_fixture_data))
def test_descriptor(fn, ad, tag_set):
    assert ad is None or ad.tags == tag_set
