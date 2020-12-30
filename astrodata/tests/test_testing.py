"""
Tests for the `astrodata.testing` module.
"""

import os
import subprocess

import astrodata
import numpy as np
import pytest
from astrodata.testing import (assert_same_class, download_from_archive,
                               get_active_git_branch)


@pytest.mark.skip("Test coverage difference - unskip later")
def test_get_active_branch_name(capsys, monkeypatch):
    """
    Just execute and prints out the active branch name.
    """

    # With tracking branch
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda *a, **k: b'(HEAD -> master, origin/master)')
    assert get_active_git_branch() == 'master'
    captured = capsys.readouterr()
    assert captured.out == '\nRetrieved active branch name:  master\n'

    # Only remote branch
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda *a, **k: b'(HEAD, origin/foo)')
    assert get_active_git_branch() == 'foo'

    monkeypatch.setattr(subprocess, 'check_output',
                        lambda *a, **k: b'(HEAD, origin/sky_sub, sky_sub)')
    assert get_active_git_branch() == 'sky_sub'

    # With other remote name
    monkeypatch.setattr(subprocess, 'check_output',
                        lambda *a, **k: b'(HEAD, myremote/foo)')
    assert get_active_git_branch() == 'foo'

    monkeypatch.setattr(subprocess, 'check_output',
                        lambda *a, **k: b'(HEAD -> master, upstream/master)')
    assert get_active_git_branch() == 'master'

    # Raise error
    def mock_check_output(*args, **kwargs):
        raise subprocess.CalledProcessError(0, 'git')
    monkeypatch.setattr(subprocess, 'check_output', mock_check_output)
    assert get_active_git_branch() is None


def test_change_working_dir(change_working_dir):
    """
    Test the change_working_dir fixture.

    Parameters
    ----------
    change_working_dir : fixture
        Custom DRAGONS fixture.
    """
    assert "astrodata/test_testing/outputs" not in os.getcwd()

    with change_working_dir():
        assert "astrodata/test_testing/outputs" in os.getcwd()

    assert "astrodata/test_testing/outputs" not in os.getcwd()

    with change_working_dir("my_sub_dir"):
        assert "astrodata/test_testing/outputs/my_sub_dir" in os.getcwd()

    assert "astrodata/test_testing/outputs" not in os.getcwd()

    dragons_basetemp = os.getenv("$DRAGONS_TEST_OUT")
    if dragons_basetemp:
        assert dragons_basetemp in os.getcwd()


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


@pytest.mark.skip("Test fixtures might require some extra-work")
def test_path_to_inputs(path_to_inputs):
    assert isinstance(path_to_inputs, str)
    assert "astrodata/test_testing/inputs" in path_to_inputs


@pytest.mark.skip("Test fixtures might require some extra-work")
def test_path_to_refs(path_to_refs):
    assert isinstance(path_to_refs, str)
    assert "astrodata/test_testing/refs" in path_to_refs
