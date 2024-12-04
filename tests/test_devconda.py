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

    # Cleanup!
    helpers.clear_conda_environment(env_name)
