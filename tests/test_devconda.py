import subprocess

import pytest


def test_create_developer_conda(
    fresh_dragons_dir,
    clear_devconda_environment,
    helpers,
):
    """Test generating a develoment environment with venv."""
    command = ["nox", "-s", "devconda"]

    devconda_result = subprocess.run(command, capture_output=True)

    helpers.validate_result(devconda_result)

    # Specified in create_venv() in noxfile.py
    expected_conda_env_name = "dragons_dev"

    assert helpers.check_for_conda_environment(
        expected_conda_env_name
    ), f"Conda env {expected_conda_env_name} not found."


@pytest.mark.parametrize("name_flag", ["-n", "--name"])
def test_create_developer_conda_with_custom_name(
    name_flag,
    fresh_dragons_dir,
    helpers,
):
    """Test creating conda dev env with custom name."""
    env_name = "custom_name"

    command = ["nox", "-s", "devconda", "--", name_flag, env_name]

    devconda_result = subprocess.run(command, capture_output=True)

    helpers.validate_result(devconda_result)

    # Specified in create_venv() in noxfile.py
    expected_conda_env_name = env_name

    assert helpers.check_for_conda_environment(
        expected_conda_env_name
    ), f"Conda env {expected_conda_env_name} not found."

    # Cleanup! This line cleans up the conda environment generated during the
    # test. The environment is automatically cleaned
    helpers.clear_conda_environment(env_name)


def test_installed_packages_conda(clean_conda_env, helpers):
    """Test that dependencies are installed as expected in the conda
    environment.
    """
    _env_name, python_bin = clean_conda_env

    # Try running a script that just contains imports for packages that
    # should now be in the environment.
    #
    # In the future, this should be a helper function or fixture.
    expected_packages = (
        "astrodata",
        "geminidr",
        "gempy",
        "gemini_instruments",
        "recipe_system",
        "numpy",
        "astropy",
        "scipy",
        "pytest",
        "gemini_obs_db",
        "gemini_calmgr",
    )

    python_imports_command = [
        str(python_bin.absolute()),
        "-c",
        "\n".join(f"import {name}" for name in expected_packages),
    ]

    python_script_result = subprocess.run(python_imports_command, capture_output=True)

    helpers.validate_result(python_script_result)
