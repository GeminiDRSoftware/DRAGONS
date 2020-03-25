#!/usr/bin/env python
"""
Tests for the `astrodata.testing` module.
"""

import os
import pytest

from astrodata.testing import download_from_archive


def test_download_from_archive_raises_ValueError_if_envvar_does_not_exists():
    with pytest.raises(ValueError):
        download_from_archive('N20180304S0126.fits', env_var='')


def test_download_from_archive_raises_IOError_if_path_is_not_accessible():
    env_var = 'MY_FAKE_ENV_VAR'
    os.environ['MY_FAKE_ENV_VAR'] = "/not/accessible/path"
    with pytest.raises(IOError):
        download_from_archive('N20180304S0126.fits', env_var=env_var)


def test_download_from_archive(monkeypatch, tmpdir):

    ncall = 0

    def mock_download(remote_url, **kwargs):
        nonlocal ncall
        ncall += 1
        fname = remote_url.split('/')[-1]
        tmpdir.join(fname).write('')  # create fake file
        return str(tmpdir.join(fname))

    monkeypatch.setattr("astrodata.testing.download_file", mock_download)
    monkeypatch.setenv("DRAGONS_TEST_INPUTS", str(tmpdir))

    # first call will use our mock function above
    fname = download_from_archive('N20170529S0168.fits', path='subdir')
    assert os.path.exists(fname)
    assert ncall == 1

    # second call will use the cache so we check that our mock function is not
    # called twice
    fname = download_from_archive('N20170529S0168.fits', path='subdir')
    assert os.path.exists(fname)
    assert ncall == 1


def test_path_to_test_data(path_to_test_data):
    assert os.path.exists(path_to_test_data)


def test_path_to_inputs(path_to_inputs):
    print(path_to_inputs)
    assert os.path.exists(path_to_inputs)


if __name__ == '__main__':
    pytest.main()
