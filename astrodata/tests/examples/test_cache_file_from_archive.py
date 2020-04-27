import pytest
from os.path import exists

from astrodata.testing import (cache_file_from_archive, path_to_inputs,
                               path_to_outputs, path_to_test_data)

filename = 'N20180304S0126.fits'


@pytest.fixture
def path(cache_file_from_archive):
    return cache_file_from_archive(filename)


def test_cache_file_from_archive_new_file(path):
    path = cache_file_from_archive(filename)
    assert isinstance(path, str)
    assert filename in path
    assert exists(path)
