import pytest
from os.path import exists

from astrodata.testing import (cache_file_from_archive, path_to_inputs,
                               path_to_outputs, path_to_test_data)


def test_cache_file_from_archive_new_file(cache_file_from_archive, capsys):

    filename = 'X20001231S0001.fits'
    path = cache_file_from_archive(filename)
    captured = capsys.readouterr()

    assert "Caching file to:" in captured.out
    assert isinstance(path, str)
    assert filename in path
    assert exists(path)

    path = cache_file_from_archive(filename)
    captured = capsys.readouterr()
    assert "Input file is cached in:" in captured.out
