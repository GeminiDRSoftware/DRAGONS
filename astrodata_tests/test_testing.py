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
        download_from_archive("N20180304S0126.fits", env_var="")


def test_download_from_archive_raises_IOError_if_path_is_not_accessible():
    env_var = "MY_FAKE_ENV_VAR"
    os.environ["MY_FAKE_ENV_VAR"] = "/not/accessible/path"
    with pytest.raises(IOError):
        download_from_archive("N20180304S0126.fits", env_var=env_var)


def test_download_from_archive(monkeypatch, tmp_path):
    ncall = 0

    def mock_download(remote_url, **kwargs):
        nonlocal ncall
        ncall += 1
        fname = remote_url.split("/")[-1]
        # Create a fake file
        with open(os.path.join(tmp_path, fname), "w+") as _:
            pass

        return str(os.path.join(tmp_path, fname))

    env_var = "TEST_NEW_CACHE"
    monkeypatch.setenv(env_var, str(tmp_path))
    monkeypatch.setattr("astrodata.testing.download_file", mock_download)

    archive_filename = "THIS_IS_A_TEST.fits"

    # In case fname is not set, we need to set it to something that will fail
    fname = ".this_does_not_exist.fits"

    try:
        # first call will use our mock function above
        fname = astrodata.testing.download_from_archive(archive_filename)
        assert os.path.exists(fname)
        assert ncall == 1

        # second call will use the cache so we check that our mock function is not
        # called twice
        fname = astrodata.testing.download_from_archive(archive_filename)
        assert os.path.exists(fname)
        assert ncall == 1

    finally:
        # Clean up the downloaded file so the test works next time.
        if os.path.exists(fname):
            os.remove(fname)


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
