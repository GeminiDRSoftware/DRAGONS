#!/usr/bin/env python
"""
Tests for the `astrodata.testing` module.
"""

import os
import pytest

from astrodata import testing


def test_download_from_archive_raises_ValueError_if_envvar_does_not_exists():
    with pytest.raises(ValueError):
        testing.download_from_archive('N20180304S0126.fits', env_var='')


def test_download_from_archive_raises_IOError_if_path_is_not_accessible():
    env_var = 'MY_FAKE_ENV_VAR'
    os.environ['MY_FAKE_ENV_VAR'] = "/not/accessible/path"
    with pytest.raises(IOError):
        testing.download_from_archive('N20180304S0126.fits', env_var=env_var)


@pytest.mark.dragons_remote_data
def test_download_from_archive():
    fname = testing.download_from_archive('N20170529S0168.fits')
    assert os.path.exists(fname)
    # make sure that download_from_archive works when the file does not exists
    os.remove(fname)
    fname = testing.download_from_archive('N20170529S0168.fits')
    assert os.path.exists(fname)


def test_path_to_test_data(path_to_test_data):
    assert os.path.exists(path_to_test_data)


def test_path_to_inputs(path_to_inputs):
    print(path_to_inputs)
    assert os.path.exists(path_to_inputs)


if __name__ == '__main__':
    pytest.main()
