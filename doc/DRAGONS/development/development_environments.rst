.. _development_environments:

Development Environments
========================

.. _nox: https://github.com/wntrblm/nox
.. _pipx: https://github.com/pypa/pipx

DRAGONS provides a common development environment, available through our `nox`_
automations.

To get started, ensure you have `nox`_ installed. We recommend using `pipx`_ to
avoid adding a package to your global environment, though you can also use
`pip` or a different platform.

To install using `pipx`_:

.. code-block:: console

   pipx install nox

To install using `pip`_:

.. code-block:: console

   pip install nox

   # Alternatively... to specify the python you're using.

   python -m pip install nox


Generating a fresh development environment
------------------------------------------

.. _venv: https://docs.python.org/3/library/venv.html
.. _conda: https://github.com/conda-forge/miniforge

We currently support two development environments, one that uses `venv`_
and another that uses `conda`_. Which you use is up to preference, though keep
the following in mind:

+ Installation of packages is the same for both environments with the exception
  of ``sextractor``. Everything else (including DRAGONS) is installed via `pip`.
+ You can pass normal `conda create` command line arguments to the `conda`
  development environment, making it slightly more configurable. Details below.


``venv`` environments
-------------------

.. warning::

   This process will overwrite existing environments and files at the local
   path `venv/`. If you have anything there you want to keep, save it before
   proceeding.

New `venv`_ environments can be generated using the ``devenv`` session:

.. code-block:: console

   nox -s devenv

This will not activate the environment for you. `venv` environments are created
in a new local directory, ``venv/``. To activate a `venv`, you run ``source
venv/bin/activate``. You'll know the environment is active when the prompt
``(dragons_venv)`` is visible on your terminal prompt. For example:

.. code-block:: console

   awesomedev@my_laptop $ source venv/bin/activate
   (dragons_venv) awesomedev@my_laptop $

Now, you will be using the correct packages and python to develop with DRAGONS.
That's it! If you decide you need a fresh environment, or update something, you
can trivially generate a new one with the exact same `nox` command.


``conda`` environments
----------------------

.. warning::

   This process will delete ``conda`` environments with the same name as the
   requested environment. By default, that is ``dragons_dev``, so if you run
   the nox command to generate a new default conda environment, it will delete
   any environment it finds named ``dragons_dev``.

   You can specify a specific name, as discussed below, and that will be
   overridden.

New `conda`_ environments can be generating using the ``devconda`` `nox`_ session:

.. code-block:: console

   nox -s devconda

You can specify arguments for ``conda create`` using a trailing ``--`` followed
by the arguments. For example, if we want to name our environment
``my_conda_env``:

.. code-block:: console

   nox -s devconda -- --name my_conda_env

By default, the environment name is ``dragons_dev``.

This script does not automatically activate your environment. To activate your
conda environment, you need to run:

.. code-block:: console

   conda activate dragons_dev

If you specified a custom name, you'll need to replace ``dragons_dev`` with
that name.
