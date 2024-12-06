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
import re

import nox


def create_venv(session: nox.Session) -> Path:
    """Create a new virtual environment using a running session."""
    default_venv_path = Path("venv/")
    default_venv_prompt = "dragons_venv"

    venv_args = ["--prompt", default_venv_prompt, "--clear", "--upgrade-deps"]

    session.run("python", "-m", "venv", default_venv_path, *venv_args)

    assert default_venv_path.exists()

    return default_venv_path


def assert_python_version(
    session: nox.Session,
    target_python: Path,
    version: tuple[int, ...],
    *,
    lowest_version: bool = True,
):
    """Assert that the python version of the target python matches a given version.

    The tuple should be te major, minor, and patch version. Any omitted values
    or values set to < 0 will act as wildcard.
    """

    python_version_str = session.run(
        str(target_python), "--version", silent=True, external=True
    )

    version_match = re.match(
        r"Python\s* ([0-9]+)\.([0-9]+)\.([0-9]+)", python_version_str
    )

    assert version_match, f"Didn't get version: {python_version_str}."

    major, minor, patch = (int(version_match.group(n)) for n in (1, 2, 3))

    for expected, found in zip([major, minor, patch], version):
        if found < 0:
            continue

        valid_version = expected == found if not lowest_version else expected >= found
        assert (
            valid_version
        ), f"Mismatched versions: {(major, minor, patch)} != {version}"


def install_dependencies(
    session: nox.Session,
    *,
    target_python: Path | None = None,
):
    """Install dependencies using a running session."""
    if target_python is None:
        target_python = Path(session.virtualenv.bin) / "python"

    assert_python_version(session, target_python, (3, 12))

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
    session.notify("install_pre_commit_hooks")


@nox.session(venv_backend="none")
def devconda(session: nox.Session):
    """Generate a new conda development environment."""
    extra_args = ["--force", "-y"]

    if all(name_flag not in session.posargs for name_flag in ["-n", "--name"]):
        env_name = "dragons_dev"
        extra_args.extend(["--name", env_name])

    else:
        name_flag_index = [c in ["-n", "--name"] for c in session.posargs].index(True)
        env_name = session.posargs[name_flag_index + 1]

    if all("python" not in arg for arg in session.posargs):
        extra_args.extend(["python=3.12"])

    session.run("conda", "create", *extra_args, *session.posargs, external=True)

    result = session.run("conda", "env", "list", silent=True, external=True)

    env_re = re.compile(r"^({env})\s+(.*)$".format(env=env_name), re.MULTILINE)

    python_path = Path(env_re.search(result).group(2)) / "bin" / "python"

    install_dependencies(session, target_python=python_path)

    session.notify("install_pre_commit_hooks")


@nox.session(venv_backend="none")
def remove_devconda_environment(session: nox.Session):
    """Remove existing dragons_dev environment generated by devconda session."""
    session.run(
        "conda", "remove", "--name", "dragons_dev", "--all", "-y", external=True
    )


@nox.session
def install_pre_commit_hooks(session: nox.Session):
    """Install pre-commit hooks; happens after devshell/devconda runs."""
    session.install("pre-commit")

    session.run("pre-commit", "install")
