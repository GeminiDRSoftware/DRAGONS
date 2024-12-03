import subprocess


def test_create_developer_conda(
    fresh_dragons_dir,
    clear_devconda_environment,
    helpers,
):
    """Test generating a develoment environment with venv."""
    command = ["nox", "-s", "devconda"]

    devenv_result = subprocess.run(command, capture_output=True)

    helpers.validate_result(devenv_result)

    # Specified in create_venv() in noxfile.py
    expected_conda_env_name = "dragons_dev"

    conda_info_result = subprocess.run(["conda", "info", "-e"], capture_output=True)

    helpers.validate_result(conda_info_result)

    stdout = conda_info_result.stdout.decode("utf-8")

    assert any(
        expected_conda_env_name in line for line in stdout.splitlines()
    ), "Env not found!"
