
.. _AstroData: https://astrodata-programmer-manual.readthedocs.io/en/v2.1.0/appendices/api_refguide.html#astrodata

.. _`built-in fixtures`: https://docs.pytest.org/en/latest/fixture.html#pytest-fixtures-explicit-modular-scalable

.. _fixtures: https://docs.pytest.org/en/latest/fixture.html

.. _fixtures_scopes : https://docs.pytest.org/en/latest/fixture.html#scope-sharing-a-fixture-instance-across-tests-in-a-class-module-or-session

.. _`matplotlib.testing`: https://matplotlib.org/3.2.1/api/testing_api.html#matplotlib-testing

.. _`numpy.testing`: https://docs.scipy.org/doc/numpy/reference/routines.testing.html

.. _pytest: https://docs.pytest.org/en/latest/

.. _tox: https://tox.readthedocs.io/en/latest/

.. _tox-conda: https://github.com/tox-dev/tox-conda

*************************************
Creating and Running Tests on DRAGONS
*************************************

This document contains some guidelines to run the existing DRAGONS tests and to
create new ones. DRAGONS testing infrastructure depends on pytest_. You can
find the Python modules containing the tests within each ``tests/`` directory
that live in the same level as the module that they are testing. Here are some
examples:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Tests for:
     - Are found in:
   * - astrodata/fits.py
     - astrodata/tests/
   * - gempy/library/astromodel.py
     - gempy/library/tests/
   * - geminidr/gmos/primitives_gmos.py
     - geminidr/gmos/tests/
   * - geminidr/gmos/recipes/ql/recipe*.py
     - geminidr/gmos/recipes/ql/tests/

Requirements
============

First of all, make sure you have all the packages required to install and run
DRAGONS.

.. todo:: Add link to requirements page.

You will also need to install pytest_, since it is the main testing suite used
in DRAGONS. Alternatively, you might want to install tox_ and the tox-conda_
extension.

Static Data
-----------

Some of DRAGONS' tests require static data that will be used as Input Data.
These tests require that you store the root path for these Input Data in the
``$DRAGONS_TEST`` environment variable. The relative path to the data depends
on the relative path of the test module within DRAGONS.

For example, if we want to find the input file for a test defined in:
``gemini_instruments/gmos/tests/test_gmos.py``, the input file would be found
inside the following directory:

.. code::

    $DRAGONS_TEST/gemini_instruments/gmos/test_gmos/inputs/

Other tests, usually called Regression Tests, require Reference Files for
comparison. These Reference Files are stored within a similar directory
structure. The only difference is the most internal directory, which should be
``**refs/**``, like this:

.. code::

    $DRAGONS_TEST/gemini_instruments/gmos/test_gmos/refs/

Here is another example:

Test module:

.. code::

   geminidr/gmos/tests/test_gmos_spect_ls_apply_qe_correction.py

New path to inputs:
.. code::

   $DRAGONS_TEST/geminidr/gmos/test_gmos_spect_ls_apply_qe_correction/inputs/

New path to refs:

.. code::

   $DRAGONS_TEST/geminidr/gmos/test_gmos_spect_ls_apply_qe_correction/refs/

This architecture allows direct relationship between the path to the data and
the test who uses it. It is important to highlight that this file management is
completely manual. There is no automatic option that would write any data into
the ``$DRAGONS_TEST`` folder.


Running Tests
=============

We can run them with the following command line in the root of the DRAGONS
repository:

.. code-block::

    $ pytest

pytest_ can be configured via a lot of command line parameters. Some
particularly useful ones are:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - -v or --verbose
     - Be more verbose.
   * - --capture=no or -s
     - Do not capture stdout (print messages in during test execution)
   * - -rs
     - Report why tests were skipped (see more)
   * - --basetemp=./temp
     - Write temporary files into the ./temp folder.

If you call pytest_ in the repository's root folder, it will run all the tests
inside DRAGONS. You can select which test(s) you want to run by directory, as
we show in the examples below:

.. code-block::

    $ pytest gempy/library

Or,

.. code-block::

    $ pytest gempy/library/tests

If you want to run a particular test within a given file, you can call pytest
with the relative path to that file followed by a double colon (::) and the
name of the test, as the example below:

.. code-block::

    $ pytest astrodata/tests/test_fits.py::test_slice


Customized Command Line Options
-------------------------------

pytest_ allows custom command line options. In DRAGONS, these options are
defined inside the ``conftest.py`` file, in the repository's root folder. Here
is a short description of each of them:

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - --dragons-remote-data
     - Enable tests that require any input data.
   * - --force-cache
     - Allows downloading input data from the archive and caching them into a
       temporary folder.
   * - --force-preprocess-data
     - Allows tests to create pre-processed data and store them into a temporary
       folder.
   * - --do-plots
     - Allows tests to plot results and save them.

Tests that require any kind of input data are normally skipped. If you want to
run them, you will have to call them using the ``--dragons-remote-data`` command
line option. These tests will fail with a ``FileNotFoundError`` error if they
cannot find the input files.

Using Tox
---------

Tests can be run directly with pytest, but this requires some work to set up the
test environment (downloading files, installing optional dependencies), and it
may not be obvious what options to use to run the different series of tests
(unit tests, integration tests, etc.).

Tox_ is a standard tool in the Python community that takes care of creating a
virtual environment (possible with conda), installing the package and its
dependencies, and running some commands.

This allows easy setup on Continuous Integration (CI) providers, like Jenkins
or GitHub Actions, and assures that the setup is the same in both of them. It
also allows developers to run tests in environments that are almost identical
to the CI server, which can be very useful for debugging.

With the current configuration, it is possible to run one of those environments:

.. code-block::

    $ pip install tox tox-conda
    $ tox -l
        py36-unit   py36-gmosls     py36-integ      py36-reg
        py37-unit   py37-gmosls     py37-integ      py37-reg
        py38-unit   py38-gmosls     py38-integ      py38-reg
        py39-unit   py39-gmosls     py39-integ      py39-reg
        codecov     check           docs-astrodata

And here are some examples to run a given environment, here running unit tests
on Python 3.7:

.. code-block::

    # simple usage:
    $ tox -e py37-unit

    # with the verbose flag, showing more detail about tox operations:
    $ tox -e py37-unit -v

    # passing additional options to pytest (arguments after the --):
    $ tox -e py37-unit -- -sv --pdb

    # specifying the environment with an environment variable:
    $ TOXENV=py37-unit tox


Writing new tests
=================

New tests for DRAGONS should use pytest_ and testing modules like
`numpy.testing`_ or `matplotlib.testing`_. In DRAGONS, we write our tests as
`part of the application code <https://docs.pytest.org/en/latest/goodpractices.html#tests-as-part-of-application-code>`_.
That means that we have a direct relation between tests and application modules.
For example:

.. code-block::

    astrodata/
        __init__.py
        factory.py
        fits.py
        (...)
        tests/
            __init__.py
            test_factory.py
            test_fits.py
            (...)

The only requirement on the test function name is that it should have a
**test_** prefix or a **_test** suffix. That means that the example below is
valid test definition:

.. code-block:: python

    def test_can_perform_task(_):
        ...
        assert task_was_performed()

In general, it is considered to be a good practice to write long and descriptive
names for test functions. Mostly because it allows faster diagnosis when some
test fails. Acronyms and test numbers usually give lesser information on why
the tests were failing. The two examples below should be **avoided**:

.. code-block:: python

    def test_cpt():
        ...
        assert task_was_performed()

    def test_1(_):
        ...
        assert task_was_performed()


Test plug-ins (fixtures)
------------------------

Pytest_ allows the creation of special functions called fixtures_. They are
usually to add custom test setup and/or finalization. Boilerplate code or code
that brings up the system to a state right before the test should usually be
written within fixtures. This is a way of isolating what is being actually
tested.

A fixture is any function containing a ``@pytest.fixture`` decorator. For
example:

.. code-block:: python

    # astrodata/tests/test_core.py
    @pytest.fixture
    def ad1():
        hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
        phu = fits.PrimaryHDU(header=hdr)
        hdu = fits.ImageHDU(data=np.ones(SHAPE), name='SCI')
        return astrodata.create(phu, [hdu])

This fixture creates a new AstroData_ object to be used in tests. Fixtures
cannot be called directly. There are several ways of plugging fixtures into
tests. DRAGONS uses the most popular one, which is adding them to the test
function argument, as the example below:

.. code-block:: python

    def test_is_astrodata(ad1):
        assert is_instance(ad1, AstroData)  # True

The ``@pytest.fixture()`` decorator can receive a scope parameter, which can
have the values of function, class, module, or session. The default scope
is ``function``. This parameter determines if the fixture should run once per
each test (``scope="function"``), once per each test file (``scope="module"``)
or once per each test session (``scope="session"``). More information on
Fixtures Scopes can be found in `this link <fixtures_scopes>`_.

Pytest_ contains several `built-in fixtures`_ that are used in DRAGONS' tests. The
most commonly used fixtures_ are:

.. list-table::
   :widths: 50 50
   :header-rows: 1

    * - capsys
      - Captures stdout and stderr messages.
    * - caplog
      - Capture and handle log messages.
    * - monkeypatch
      - Modify objects and environment.
    * - tmp_path_factory
      - Returns a function used to access a temporary folder unique for each
        test session.
    * - request
      - Passes information from the test function to within the fixture being
        called.

Pytest_ fixtures_ are modular since they can be used by fixtures_. This allowed
the creation of custom fixtures_ for the DRAGONS Testing Suite.

For example, the ``astrodata.testing`` module contains several fixtures that
handle reading/writing/caching data. These fixtures are used directly in tests
or inside other fixtures just like fixtures are used inside tests (as function
arguments).

The diagram below shows the hierarchical structure of the main fixtures used for
data handling in DRAGONS:












