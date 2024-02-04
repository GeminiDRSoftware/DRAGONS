import pytest
import astrodata
import astrodata.testing
import gemini_instruments
import os

THIS_DIR = os.path.dirname(__file__)

from .lut_descriptors import fixture_data as descriptors_fixture_data
# Keep separate because GHOST has non-raw files
from .ghost_lut_descriptors import fixture_data as ghost_descriptors_fixture_data


@pytest.mark.parametrize("instr,filename,descriptor,value",
                         ([*k]+[*vv] for k, v in (list(descriptors_fixture_data.items()) +
                                                  list(ghost_descriptors_fixture_data.items()))
                          for vv in v))
def test_descriptor(instr, filename, descriptor, value):
    path_to_test_data = os.getenv("DRAGONS_TEST")
    if path_to_test_data is None:
        pytest.skip('Environment variable not set: $DRAGONS_TEST')

    path_to_test_data = os.path.expanduser(path_to_test_data).strip()
    path_to_files = ['gemini_instruments', 'test_astrodata_tags',
                     'inputs']
    path_to_test_data = os.path.join(path_to_test_data, *path_to_files)

    if '_' in filename:
        filepath = os.path.join(path_to_test_data, filename)
        try:
            ad = astrodata.open(filepath)
        except FileNotFoundError:
            raise
            pytest.skip(f"{filename} not found")
    else:
        ad = astrodata.open(astrodata.testing.download_from_archive(filename))

    method = getattr(ad, descriptor)
    if value is None:
        assert method() is None
    else:
        mvalue = method()
        if float in (type(value), type(mvalue)):
            assert abs(mvalue - value) < 0.0001
        else:
            assert value == mvalue
