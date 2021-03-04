import os
import subprocess

import pytest
from pytest_dragons.plugin import get_active_git_branch


def test_change_working_dir(change_working_dir):
    """
    Test the change_working_dir fixture.

    Parameters
    ----------
    change_working_dir : fixture
        Custom DRAGONS fixture.
    """
    assert "pytest_dragons/test_plugin/outputs" not in os.getcwd()

    with change_working_dir():
        assert "pytest_dragons/test_plugin/outputs" in os.getcwd()

    assert "pytest_dragons/test_plugin/outputs" not in os.getcwd()

    with change_working_dir("my_sub_dir"):
        assert "pytest_dragons/test_plugin/outputs/my_sub_dir" in os.getcwd()

    assert "pytest_dragons/test_plugin/outputs" not in os.getcwd()

    dragons_basetemp = os.getenv("$DRAGONS_TEST_OUT")
    if dragons_basetemp:
        assert dragons_basetemp in os.getcwd()


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


@pytest.mark.skip("Test fixtures might require some extra-work")
def test_path_to_inputs(path_to_inputs):
    assert isinstance(path_to_inputs, str)
    assert "astrodata/test_testing/inputs" in path_to_inputs


@pytest.mark.skip("Test fixtures might require some extra-work")
def test_path_to_refs(path_to_refs):
    assert isinstance(path_to_refs, str)
    assert "astrodata/test_testing/refs" in path_to_refs
