"""Tests for the development environment and its setup."""

from pathlib import Path
import subprocess


def test_generating_test_environment(
    tmp_path,
    monkeypatch,
    fresh_dragons_dir,
    helpers,
):
    """Test developer venv environment generation."""
    assert Path("noxfile.py").exists()

    # Generate the developer environment.
    command = ["nox", "-s", "devenv"]

    result = subprocess.run(command, capture_output=True)

    helpers.validate_result(result)

    stdout, stderr = (out.decode("utf-8") for out in (result.stdout, result.stderr))

    assert "warning" not in stdout.casefold(), "Warning appeared in stdout"
    assert "warning" not in stderr.casefold(), "Warning appeared in stderr"


def test_create_developer_venv(
    fresh_dragons_dir,
    helpers,
):
    """Test generating a develoment environment with venv."""
    command = ["nox", "-s", "devenv"]

    devenv_result = subprocess.run(command, capture_output=True)

    helpers.validate_result(devenv_result)

    # Specified in create_venv() in noxfile.py
    expected_venv_path = Path("venv/")
    expected_prompt = "dragons_venv"

    assert expected_venv_path.exists()
    assert (expected_venv_path / "bin" / "python").exists()
    assert (expected_venv_path / "bin" / "pip").exists()

    # Check the venv's prompt.
    venv_activate_path = expected_venv_path / "bin" / "activate"
    assert f"{expected_prompt}" in venv_activate_path.read_text()


def test_recreate_developer_venv(
    fresh_dragons_dir,
    helpers,
):
    """Test re-generating a develoment environment with venv."""
    command = ["nox", "-s", "devenv"]

    for _ in range(2):
        devenv_result = subprocess.run(command, capture_output=True)

        helpers.validate_result(devenv_result)

        # Specified in create_venv() in noxfile.py
        expected_venv_path = Path("venv/")
        expected_prompt = "dragons_venv"

        assert expected_venv_path.exists()
        assert (expected_venv_path / "bin" / "python").exists()
        assert (expected_venv_path / "bin" / "pip").exists()

        # Check the venv's prompt.
        venv_activate_path = expected_venv_path / "bin" / "activate"
        assert f"{expected_prompt}" in venv_activate_path.read_text()


def test_installed_packages(fresh_dragons_dir, helpers):
    """Test that dependencies are installed as expected.

    Since we assume here that the dependencies are installed via other
    channels (pip installing dragons), this only tests the existence of
    specific, expected packages.
    """

    command = ["nox", "-s", "devenv"]

    venv_python_path = Path("venv") / "bin" / "python"

    devenv_result = subprocess.run(command, capture_output=True)

    helpers.validate_result(devenv_result)

    # Try running a script that just contains imports for packages that
    # should now be in the environment.
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
    )

    python_imports_command = [
        str(venv_python_path),
        "-c",
        "\n".join(f"import {name}" for name in expected_packages),
    ]

    python_imports_result = subprocess.run(python_imports_command, capture_output=True)

    helpers.validate_result(python_imports_result)
