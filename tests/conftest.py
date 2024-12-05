"""Configurations and fixtures for package-level testing.

Most of this is for development testing. Be aware that using some of these
fixtures can add significant time to test runs, as they are hefty (but
necessary) operations.
"""

import os
from contextlib import chdir
from pathlib import Path
import re
import shutil
import subprocess

import pytest


class Helpers:
    @staticmethod
    def validate_result(
        result: subprocess.CompletedProcess,
        expected_returncode: int = 0,
    ):
        """Raises an exception if the returncode is not as expected."""
        if not result.returncode == 0:
            divider = 80 * "-"
            stdout = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8")
            message = (
                f"    Command: {result.args}\n"
                f"    stdout:\n{divider}\n{stdout}"
                f"    stderr:\n{divider}\n{stderr}"
            )

            raise Exception(message)

    @staticmethod
    def clear_conda_environment(env_name: str = "dragons_dev", *, strict: bool = False):
        """Clear a conda environment, raise exception if it exists after."""

        conda_remove_command = [
            "conda",
            "remove",
            "--name",
            env_name,
            "--all",
            "-y",
        ]

        if strict and Helpers.check_for_conda_environment(env_name):
            raise Exception(f"Could not find env with name: {env_name}")

        result = subprocess.run(conda_remove_command, capture_output=True)

        Helpers.validate_result(result)

        if Helpers.check_for_conda_environment(env_name):
            raise Exception(f"Could not remove env: {env_name}")

    @staticmethod
    def check_for_conda_environment(env_name: str) -> bool:
        """Return True if conda environment is found locally."""
        fetch_conda_envs_command = ["conda", "info", "-e"]
        fetch_envs_result = subprocess.run(
            fetch_conda_envs_command, capture_output=True
        )

        env_match_re = re.compile(r"^\b({env})\b.*$".format(env=env_name), re.MULTILINE)

        Helpers.validate_result(fetch_envs_result)

        stdout = fetch_envs_result.stdout.decode("utf-8")

        return bool([match for match in env_match_re.finditer(stdout)])

    @staticmethod
    def get_conda_environments() -> list[str]:
        """Determine the names of all conda environments current available."""
        return [k for k in Helpers.get_conda_python_paths()]

    @staticmethod
    def get_conda_python_paths() -> dict[str, Path]:
        """Get conda envs and their corresponding python paths."""
        command = ["conda", "env", "list"]

        result = subprocess.run(command, capture_output=True)

        Helpers.validate_result(result)

        stdout = result.stdout.decode("utf-8")

        paths = {}

        for line in (l.strip() for l in stdout.splitlines()):
            if not line or line[0] == "#":
                continue

            cols = line.split()
            if len(cols) >= 2:
                python_bin = Path(cols[1]) / "bin" / "python"

                if "envs" not in str(python_bin) or not python_bin.exists():
                    continue

                paths[cols[0]] = python_bin

        return paths


@pytest.fixture(scope="session")
def helpers() -> Helpers:
    return Helpers()


@pytest.fixture(scope="session")
def session_DRAGONS(tmp_path_factory, helpers) -> Path:
    """Create a clean copy of the repo.

    The steps for this fixture are:
    1. Creates a new directory caching a clean DRAGONS version, if
       it doesn't exist.
    2. Clones the repository into the temporary dir.
    3. Checks out the same branch that this repo is running on, if available.
    4. Return path to the session DRAGONS.

    WARNING: Do not modify the sssion DRAGONS. If you need a clean dragons
    dir, use the ``fresh_dragons_dir`` fixture.
    """
    tmp_path = tmp_path_factory.mktemp("cached_DRAGONS")

    dragons_path = tmp_path / "DRAGONS"
    local_repo_path = Path(__file__).parent.parent

    # Cloning the local repo
    clone_command = [
        "git",
        "clone",
        str(local_repo_path.absolute()),
        str(dragons_path.absolute()),
    ]

    branch_command = ["git", "branch", "--show-current"]

    with chdir(tmp_path):
        result = subprocess.run(clone_command, capture_output=True)

    try:
        helpers.validate_result(result)

    except Exception as err:
        message = "Could not clone dragons repo."
        raise Exception(message) from err

    with chdir(dragons_path):
        branch_result = subprocess.run(
            branch_command,
            capture_output=True,
        )

        helpers.validate_result(branch_result)

    return dragons_path


@pytest.fixture(scope="function")
def fresh_dragons_dir(
    tmp_path,
    monkeypatch,
    session_DRAGONS,
    helpers,
) -> Path:
    """Copy a new unmodified DRAGONS dir to a tempoary directory.

    This is meant to be used with development environment tests, not
    other DRAGONS tests (without good reason).

    This will be periodically cleaned up by pytest, but may store <~10 clones
    during normal execution.
    """
    dragons_dir = tmp_path / "DRAGONS"

    assert not dragons_dir.exists()

    shutil.copytree(session_DRAGONS, dragons_dir)

    monkeypatch.chdir(dragons_dir)

    return dragons_dir


@pytest.fixture()
def clear_devconda_environment(helpers) -> bool:
    """Clear the conda development environment, if it exists."""
    helpers.clear_conda_environment()

    yield

    # Cleanup
    helpers.clear_conda_environment()


@pytest.fixture()
def clean_conda_env(helpers, fresh_dragons_dir) -> tuple[str, Path]:
    """Create a clean conda environment for the test, returns name and path to
    the python binary.
    """
    devconda_command = ["nox", "-s", "devconda"]

    helpers.clear_conda_environment()

    prev_conda_envs = helpers.get_conda_environments()

    result = subprocess.run(devconda_command, capture_output=True)

    helpers.validate_result(result)

    new_conda_envs = helpers.get_conda_environments()

    env_diffs = [k for k in new_conda_envs if k not in prev_conda_envs]

    assert len(env_diffs) > 0, "No new environments created."

    assert (
        len(env_diffs) == 1
    ), "Multiple new environments detected, This is not thread safe!"

    new_env = env_diffs[0]
    env_pythons = helpers.get_conda_python_paths()

    yield [new_env, env_pythons[new_env]]

    helpers.clear_conda_environment()
