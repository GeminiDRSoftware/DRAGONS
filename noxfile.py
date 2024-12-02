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

1. Is your automation meant to perform setup or teardown for a test? If so,
   try to do it in pytest.

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

import nox


@nox.session(venv_backend="venv")
def devenv(session: nox.Session):
    """Generate a new development environment."""
    session.run("python", "-m", "venv", *session.posargs)
