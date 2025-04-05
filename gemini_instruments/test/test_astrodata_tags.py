import pytest
import astrodata
import astrodata.testing
import gemini_instruments
import os

from .lut_tags import fixture_data as tags_fixture_data


@pytest.mark.dragons_remote_data
@pytest.mark.parametrize("instr,filename,tag_set",
                         ([*k]+[v] for k, v in tags_fixture_data.items()))
def test_descriptor(instr, filename, tag_set):
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
            ad = astrodata.from_file(filepath)
        except FileNotFoundError:
            pytest.skip(f"{filename} not found")
    else:
        ad = astrodata.from_file(astrodata.testing.download_from_archive(filename))

    assert ad.tags == set(tag_set)
