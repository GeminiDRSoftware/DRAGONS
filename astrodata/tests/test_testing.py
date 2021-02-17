"""
Tests for the `astrodata.testing` module.
"""

import os

import astrodata
import numpy as np
import pytest
from astrodata.testing import assert_same_class, download_from_archive


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
    monkeypatch.setenv("DRAGONS_TEST", str(tmpdir))

    # first call will use our mock function above
    fname = download_from_archive('N20170529S0168.fits')
    assert os.path.exists(fname)
    assert ncall == 1

    # second call will use the cache so we check that our mock function is not
    # called twice
    fname = download_from_archive('N20170529S0168.fits')
    assert os.path.exists(fname)
    assert ncall == 1


def test_assert_most_close():
    from astrodata.testing import assert_most_close
    x = np.arange(10)
    y = np.arange(10)
    assert_most_close(x, y, 1)

    y[0] = -1
    assert_most_close(x, y, 1)

    with pytest.raises(AssertionError):
        y[1] = -1
        assert_most_close(x, y, 1)


def test_assert_most_equal():
    from astrodata.testing import assert_most_equal
    x = np.arange(10)
    y = np.arange(10)
    assert_most_equal(x, y, 1)

    y[0] = -1
    assert_most_equal(x, y, 1)

    with pytest.raises(AssertionError):
        y[1] = -1
        assert_most_equal(x, y, 1)


def test_assert_same_class():
    ad = astrodata.create({})
    ad2 = astrodata.create({})
    assert_same_class(ad, ad2)

    with pytest.raises(AssertionError):
        assert_same_class(ad, np.array([1]))
