"""Automations that run in isolated python environments.

To see a list of available commands, run::

    nox -l

It is expected that you have ``nox`` installed on your machine. Other than
that, all other installation for dependencies is covered by the automations in
this file.

Modifying or adding to the noxfile
==================================

If you find a good use case for automation, great! Before adding it to this
already large file, please consider the following questions about the scope of
this file:

1. Is your automation meant to perform setup or teardown for a test? If so, try to do it in pytest.

   + If you are working with something like ``devpi``, where it's important that
     some packages be isolated, then this is the appropriate place for a change.

2. Are you trying to provision a resource, such as asking for more workers to
   split up automation tasks? If so, use something else.

   + ``nox`` is meant for automations at the scope of a single Python binary.
     This means that if you want more computing power, you need to go above
     ``nox``.

3. Are you planning to generate test files, or run test scripts? If so, use
   ``pytest`` and in one of the relevant ``./*/tests/`` directories.

4. Is this code that will not need to be modified often? The main reason this
   file is allowed to be this big is that, ideally, one should not be modifying
   nox sessions regularly. If the behavior you require will need manual
   editing, rethink it or put it elsewhere as a script, please.

For more information, see the DRAGONS developer documentation.
"""

import tomllib
from pathlib import Path

import nox


def create_venv(session: nox.Session) -> Path:
    """Create a new virtual environment using a running session."""
    default_venv_path = Path("venv/")
    default_venv_prompt = "dragons_venv"

    venv_args = ["--prompt", default_venv_prompt, "--clear", "--upgrade-deps"]

    session.run("python", "-m", "venv", default_venv_path, *venv_args)

    assert default_venv_path.exists()

    return default_venv_path


def install_dependencies(
    session: nox.Session,
    *,
    target_python: Path | None = None,
):
    """Install dependencies using a running session."""
    if target_python is None:
        target_python = Path(session.virtualenv.bin) / "python"

    # Install development dependencies from pyproject.toml
    # TODO: This must be changed when the dependency manager is changed.
    pyproject_toml_path = Path("pyproject.toml")
    with pyproject_toml_path.open("rb") as infile:
        pyproject_toml_contents = tomllib.load(infile)

    dev_dependencies = pyproject_toml_contents["development"]["dependencies"]

    session.run(
        str(target_python),
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        external=True,
    )

    session.run(
        str(target_python),
        "-m",
        "pip",
        "install",
        *dev_dependencies,
        external=True,
    )


@nox.session(venv_backend="venv")
def devenv(session: nox.Session):
    """Generate a new venv development environment."""
    venv_path = create_venv(session)
    venv_python = venv_path / "bin" / "python"
    install_dependencies(session, target_python=venv_python)


@nox.session(venv_backend="conda")
def devconda(session: nox.Session):
    """Generate a new conda development environment."""
    extra_args = ["--force", "-y"]

    if all(name_flag not in session.posargs for name_flag in ["-n", "--name"]):
        extra_args.extend(["--name", "dragons_dev"])

    session.run("conda", "create", *extra_args, *session.posargs)
