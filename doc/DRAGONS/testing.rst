.. testing.rst

.. _AstroData: https://astrodata-programmer-manual.readthedocs.io/en/v2.1.0/appendices/api_refguide.html#astrodata
.. _command-line: https://docs.pytest.org/en/latest/usage.html
.. _fixture: https://docs.pytest.org/en/latest/fixture.html
.. _fixtures: https://docs.pytest.org/en/latest/fixture.html
.. _pip: https://pip.pypa.io/en/stable/
.. _PyTest: https://docs.pytest.org/en/stable/
.. _tox: https://tox.readthedocs.io/en/latest/

.. _create_and_run_tests:


Create and Run Tests
====================

This document contains some guidelines to run the existing DRAGONS tests and to
create new ones. DRAGONS testing infrastructure depends on PyTest_. You can
find the tests inside the :code:`tests/` directory at the same level of the
module that they are testing. For example:

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
------------

At first, you should be able to run tests using only pytest_. Most of the other
requirements that you might need are satisfied by the DRAGONS requirements
itself.

However, you **do** need to install DRAGONS if you want to run tests. Even if
you are testing the source code itself. This is the easiest (if not the only)
way to make the DRAGONS' plugin for pytest_ that contains several customizations
for DRAGONS tests available for us.

.. todo: Point to the standard DRAGONS installation.

You can install DRAGONS from its source code in your working environment easily
using pip_ with the commands in a terminal. The commands below will clone the
latest version of DRAGONS from GitHub to your computer and install it:

.. code-block:: bash

    $ mkdir /path/to/dragons/   # Create folder for DRAGONS
    $ git -C /path/to/dragons/ clone https://github.com/GeminiDRSoftware/DRAGONS.git
    $ cd /path/to/dragons/  # Enter DRAGONS folder
    $ pip install -e .  # Install with PiP

The :code:`$` symbol simply represents command line typed into a terminal.
The :code:`-e` flag is optional, but very handy. It is used to create a link to
our source code inside the path to the Python installed libraries. This way,
if you make changes to your source code, these changes will be reflected in the
installed DRAGONS.

In addiction to installing DRAGONS, you might need to provide input data to the
tests. The root path of the input data should be stored in the ``$DRAGONS_TEST``
environment variable, which can be set with:

.. code-block:: bash

    $ export DRAGONS_TEST="/path/to/your/input/files"

or by adding the command above to your :code:`.bashrc`, :code:`.bash_profile`,
or equivalent configuration file.

The relative path to the data depends on the relative path of the test module
within DRAGONS. For example, if we want to find the input file for a test
defined in :code:`gemini_instruments/gmos/tests/test_gmos.py`, the input file
would be found inside
:code:`$DRAGONS_TEST/gemini_instruments/gmos/test_gmos/inputs/`.

Other tests, usually called regression tests, require some reference files for
comparison. These reference files are stored within a similar directory
structure. The only difference is the most internal directory, which should be
:code:`refs/`, like this:
:code:`$DRAGONS_TEST/gemini_instruments/gmos/test_gmos/refs/`.

Here is another example:

.. list-table::
   :widths: 25 75

   * - Test module:
     - geminidr/gmos/tests/test_gmos_spect_ls_apply_qe_correction.py
   * - Path to inputs:
     - $DRAGONS_TEST/geminidr/gmos/test_gmos_spect_ls_apply_qe_correction/inputs/
   * - Path to refs:
     - $DRAGONS_TEST/geminidr/gmos/test_gmos_spect_ls_apply_qe_correction/refs/

This architecture allows a direct relationship between the path to the data and
the test that uses it. It is important to highlight that this file management is
completely manual. There is no automatic option that would write any data into
the :code:`$DRAGONS_TEST` folder.

Any automatic option that cache or preprocess data will only write them into
temporary folders.

Using automatic options to cache and/or preprocess data might compromise the
test itself since the input data might not be the same as it should be.


Running Tests with PyTest
-------------------------

We can run existing tests using the following command-line in the root of the
DRAGONS repository:

.. code-block::

    $ pytest

pytest_ can be configured via a lot of `command-line`_ parameters. Some
particularly useful ones are:

.. list-table::
   :widths: 25 75

   * - ``-v`` or ``--verbose``
     - Be more verbose.
   * - ``--capture=no`` or ``-s``
     - Do not capture stdout (print messages in during test execution).
   * - ``-rs``
     - Report why tests were skipped (see `more <https://docs.pytest.org/en/latest/usage.html#detailed-summary-report>`_)
   * - ``--basetemp=./temp``
     - Write temporary files into the :code:`./temp` folder.

Calling pytest_ in the repository's root folder will run all the tests inside
DRAGONS. You can select which test(s) you want to run by package (directory), as
we show in the examples below:

.. code-block:: bash

    $ pytest gempy/library/

Or,

.. code-block:: bash

    $ pytest gempy/library/tests/

If you want to run a particular test within a given module (file), you can call
pytest with the relative path to that file followed by a double colon (::) and
the name of the test, as the example below:

.. code-block:: bash

    $ pytest astrodata/tests/test_fits.py::test_slice

Customized Command-Line Options
-------------------------------

pytest_ allows custom command-line options. In DRAGONS, these options are
defined inside the :code:`pytest_dragons/plugin.py` file, in the repository's
root folder. Here is a short description of each of them:

.. list-table::
   :widths: 25 75

   * - ``--dragons-remote-data``
     - Enable tests that require any input data.
   * - ``--force-cache``
     - Allows downloading input data from the archive and caching them into a temporary folder.
   * - ``--interactive``
     - Runs tests that have some interactive component.

Tests that require any kind of input data are normally skipped. If you want to
run them, you will have to call them using the :code:`--dragons-remote-data`
command-line option. These tests will fail with a :code:`FileNotFoundError` if
they cannot find the input files.


Running Tests with Tox
----------------------

Tests can be run directly with pytest_, but this requires some work to set up
the test environment (downloading files, installing optional dependencies), and
it may not be obvious what options to use to run the different series of tests
(unit tests, integration tests, etc.).

Tox_ is a standard tool in the Python community that takes care of creating a
virtualenv (possible with conda), installing the package and its dependencies,
and running some commands.

This allows easy setup on Continuous Integration (CI) providers, like
Jenkins or GitHub Actions, and assures that the setup is the same in both of
them.

It also allows developers to run tests in environments that are almost identical
to the CI server, which can be very useful for debugging.

With the current configuration, it is possible to run one of those environments:

.. code-block:: bash

   $ pip install tox tox-conda
   $ cd /path/to/dragons/
   $ tox -l

   py36-unit    py37-unit    py38-unit    py39-unit    codecov
   py36-gmosls  py37-gmosls  py38-gmosls  py39-gmosls  check
   py36-integ   py37-integ   py38-integ   py39-integ   docs-astrodata
   py36-reg     py37-reg     py38-reg     py39-reg

And here are some examples to run a given environment, here running unit tests
on Python 3.7:

.. code-block:: bash

   # simple usage:
   $ tox -e py37-unit

   # with the verbose flag, showing more detail about tox operations:
   $ tox -e py37-unit -v

   # passing additional options to pytest (arguments after the --):
   $ tox -e py37-unit -- -sv --pdb

   # specifying the environment with an environment variable:
   $ TOXENV=py37-unit tox


Pinpointing Tests
-----------------

It is important to mention that the calls when using PyTest or Tox are slightly
different. PyTest, by default, will test the source code itself. Our Tox settings
are configure to use PyTest on installed code instead. This a slight difference
but might have major impact on how to call tests and how they behave.

If you want to run a test inside a module using PyTest, you can run the following
command:

.. code-block:: bash

   $ pytest geminidr/gmos/tests/spect/test_find_source_apertures.py

With Tox, you must specify the module name instead:

.. code-block:: bash

   $ tox -e py37-gmosls -- geminidr.gmos.tests.spect.test_find_source_apertures

Remember that the ``-e py37-gmosls`` is simply the name of a Tox environment
that run tests marked with `@pytest.mark.gmosls`.

If we want to run a single test inside that module, we need to append
:code:`::test_...` after the module name. Something like this:

.. code-block:: bash

   $ pytest geminidr/gmos/tests/spect/test_find_source_apertures.py::test_find_apertures_with_fake_data

To run the test with PyTest. Or:

.. code-block:: bash

   $ tox -e py37-gmosls -- geminidr.gmos.tests.spect.test_find_source_apertures::test_find_apertures_with_fake_data

To run the test with Tox.


Writing new tests
=================

New tests for DRAGONS should use pytest_ and testing modules like
`numpy.testing <https://docs.scipy.org/doc/numpy/reference/routines.testing.html>`_
or `matplotlib.testing <https://matplotlib.org/3.2.1/api/testing_api.html#matplotlib-testing>`_.

In DRAGONS, we write our tests as
`part of the application code <https://docs.pytest.org/en/latest/goodpractices.html#tests-as-part-of-application-code>`_.
This means that we have a direct relation between tests and application modules.
For example:

::

    + astrodata/
    |--- __init__.py
    |--- factory.py
    |--- fits.py
    |--- (...)
    |---+ tests/
    |   |--- __init__.py
    |   |--- test_factory.py
    |   |--- test_fits.py
    |   |--- (...)


The only requirement on the test function name is that it should have a **test_**
prefix or a **_test** suffix. That means that the example below is a valid test
definition:

.. code-block:: python

    def test_can_perform_task():
        ...
        assert task_was_performed()


In general, writing a long descriptive name containing the function that it is
testing and what it is supposed to do is considered a good practice. Mostly
because it allows faster diagnosis when some test fails. Acronyms and test
numbers usually give lesser information on why the tests were failing. Please,
**avoid** the two examples below:

.. code-block:: python

    def test_cpt():
        ...
        assert task_was_performed()


    def test_1():
        ...
        assert task_was_performed()


Test plug-ins (fixtures)
------------------------

PyTest_ allows the creation of special functions called fixtures_. They are
usually used to add custom test setup and/or finalization. Boilerplate code or
code that brings up the system to a state right before the test should usually
be written within fixtures_. This is a way of isolating what is being actually
tested. It is also a practical way to generate test data which can be used in
multiple tests.

A fixture_ is any function containing a :code:`@pytest.fixture` decorator. For
example:

.. code-block:: python
   :caption: astrodata/tests/test_core.py

    @pytest.fixture
    def ad():
        hdr = fits.Header({'INSTRUME': 'darkimager', 'OBJECT': 'M42'})
        phu = fits.PrimaryHDU(header=hdr)
        hdu = fits.ImageHDU(data=np.ones(SHAPE), name='SCI')
        return astrodata.create(phu, [hdu])

This fixture_ creates a new AstroData_ object to be used in tests. Fixtures_
cannot not be called directly. There are several ways of plugging fixtures into
tests. DRAGONS uses the most popular one, which is adding them to the test
function argument, as the example below:

.. code-block:: python

    def test_is_astrodata(ad):
        assert is_instance(ad, AstroData)  # True

The :code:`@pytest.fixture()` decorator can receive a :code:`scope` parameter,
which can have the values of :code:`function`, :code:`class`, :code:`module`, or
:code:`session`. The default scope is :code:`function`. This parameter
determines if the fixture should run once per each test
(:code:`scope="function"`), once per each test file (:code:`scope="module"`) or
once per each test session (:code:`scope="session"`). More information on
Fixtures Scopes can be found
`in this link <https://docs.pytest.org/en/latest/fixture.html#scope-sharing-a-fixture-instance-across-tests-in-a-class-module-or-session>`_.

PyTest_ contains several
`built-in fixtures <https://docs.pytest.org/en/latest/fixture.html#pytest-fixtures-explicit-modular-scalable>`_
that are used in DRAGONS' tests. The most commonly used fixtures are:

.. list-table::
   :widths: 25 50

   * - capsys
     - Captures stdout and stderr messages.
   * - caplog
     - Capture and handle log messages.
   * - monkeypatch
     - Modify objects and environment.
   * - tmp_path_factory
     - Returns a function used to access a temporary folder unique for each test session.
   * - request
     - Passes information from the test function to within the fixture being called.

PyTest fixtures are modular since they can be used by fixtures. This allowed the
creation of custom fixtures for the DRAGONS Testing Suite. All our custom
fixtures now live inside the ``pytest_dragons/plugin.py`` module, where they are
imported from ``pytest_dragons/fixtures.py``.

Here is a very brief description of the fixtures defined in this plugin module:

.. list-table::
   :widths: 25 50

   * - change_working_dir
     - Context manager that allows easily changing working directories.
   * - path_to_inputs
     - Absolute directory path to local static input data.
   * - path_to_common_inputs
     - Absolute directory path to local static input data that is required by multiple tests.
   * - path_to_refs
     - Absolute directory path to local static reference data.
   * - path_to_outputs
     - Absolute directory path to temporary or static output data.


PyTest Configuration File
-------------------------

Most of `pytest`_'s setup and customization happens inside a special file named
:code:`conftest.py`. This file might contain fixtures that can be used in tests
without being imported and custom command-line options. Before moving towards the
``pytest_dragons`` plugin, this was how DRAGONS had all its custom setup. You can
still create a per-package :code:`conftest.py` file with specific behavior but
we invite you to discuss with us if the required new functionality might be
incorporated to the project level plugin.


Parametrization
---------------

Pytest_ allows `parameterization of tests and fixtures <https://docs.pytest.org/en/latest/parametrize.html#parametrizing-fixtures-and-test-functions>`_.
The following sections show how to parametrize tests in three different ways.
It is important to notice that mixing these three kinds of parametrization is
allowed and might lead to a matrix of parameters. This might or not be the
desired effect, so proceed with caution.


Parametrizing tests
^^^^^^^^^^^^^^^^^^^

Tests can be directly parametrized using the :code:`@pytest.mark.parametrize`
decorator.

.. code-block:: python

   list_of_parameters = [
    ('apple', 3),
    ('orange', 2),
   ]

   @pytest.mark.parametrize("fruit,number", list_of_parameters)
   def test_number_of_fruits(fruit, number):
      assert fruit in ['apple', 'banana', 'orange']
      assert isinstance(number, int)

The example above shows that parametrize's first argument should be a string
containing the name of parameters of the test. The second argument should be a
list (dictionaries and sets **do not** work) containing tuples of lists with
the same number of elements as the number of parameters.


Parametrizing fixtures
^^^^^^^^^^^^^^^^^^^^^^

If your input parameters have to pass through a fixture (e.g., the parameter is
a file name and the fixture reads and returns this file), you can parametrize
the fixture itself directly.

The example below shows how to parametrize a custom fixture using the
:code:`request` fixture, which is a built-in fixture in pytest_ that holds
information about the fixture and the test themselves. Line 08 shows how to pass
the parameter to the fixture using the :code:`request.param` variable.

.. code-block:: python

   input_files = [
    'N20001231_S001.fits',
    'N20001231_S002.fits',
   ]

   @pytest.fixture(params=input_files)
   def ad(request):
      filename = request.param
      return astrodata.open(filename)

   def test_is_astrodata(ad):
      assert isinstance(ad, AstroData)

If you parametrize more than one fixture, you will end up with a matrix of test
cases.


Indirect Fixture Parametrization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, it is possible to parametrize tests and pass these parameters to a
fixture using :code:`indirect=True` argument in :code:`@pytest.mark.parametrize`.
This is only required when you want to have a single list of parameters and some
of these parameters need to pass through a fixture. Here is an example:

.. code-block:: python

   pars = [
       # Input File, Expected Value
       ('N20001231_S001.fits', 5),
       ('N20001231_S002.fits', 10),
   ]

   @pytest.fixture
   def ad(request):
       filename = request.param
       return astrodata.open(filename)

   @pytest.fixture
   def numeric_par(request):
       return request.param

   @pytest.mark.parametrize("ad,numeric_par", pars, indirect=True)
   def test_function_returns_int(ad, numeric_par):
       assert function_returns_int(ad) == numeric_par

This method allows passing one of the input parameters to a fixture while
preventing the undesired creation of a matrix of test cases. It is also useful
because the test reports will show tests with the parameter value instead of
some cryptic value. Note that, when using :code:`indirect=True`, every parameter
has to be represented as a fixture, even if it simply forwards the parameter
value.


Creating inputs for tests
-------------------------

Most of the tests for primitives and recipes require partially-processed data.
This data must be static and, ideally, should be recreated only in rare cases.
This data should be created using a recipe that lives in the same file as the
test. For now, all the recipes that create inputs should start with
:code:`create_`. Inputs for these recipes can be defined within the function
itself or can come from variables defined in the outer scope.

These functions can be called using the :code:`--create-inputs` command option,
which is implemented simply:

.. code-block:: python

   if __name__ == '__main__':
    import sys
    if "--create-inputs" in sys.argv[1:]:
        create_inputs_for_my_test()
    else:
        pytest.main()


Ideally, these recipes should write the created inputs inside
:code:`./dragons_tests_inputs/` folder following the same directory structure
inside ``$DRAGONS_TEST`` in order to allow easy, but still manual,
synchronization.


Test markers
------------

Pytest also allows custom markers that can be used to select tests or to add
custom behaviour. These custom markers are applied using
:code:`@pytest.mark.(mark_name)`, where (mark_name) is replaced by any values in
the table below:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Marker Name
     - Description
   * - ``dragons_remote_data``
     - Tests that require data that can be downloaded from the Archive. Require ``--dragons-remote-data`` and ``$DRAGONS_TEST`` to run. It downloads and caches data.
   * - ``integration_test``
     - Long tests using ``Reduce(...)``. Only used for test selection.
   * - ``interactive``
     - For tests that requires (user) interaction and should be skipped by any Continuous Integration service.
   * - ``gmosls``
     - GMOS Long-slit Tests. Only used for test selection.
   * - ``preprocessed_data``
     - Tests that require preprocessed data. If input files are not found, they raise a FileNotFoundError. If you need to create inputs, see Create inputs for tests above.
   * - ``regression``
     - Tests that will compare output data with reference data.
   * - ``slow``
     - Slow tests. Only used for test selection.

These are the official custom markers that now live inside DRAGONS. Other custom
markers might be found and those should be removed. Any new custom marker needs
to be properly registered in the :code:`setup.cfg` file.

Examples
========

In the following example, :code:`processed_flat` is a fixture (function) that
reduces a FLAT and returns the processed AD object. :code:`reference_ad` is a
fixture (function) that returns a function and that function returns the AD
object with the given filename from the :code:`path_to_refs`.

.. code-block:: python
   :caption: geminidr/gmos/recipes/ql/tests/test_make_processed_flat.py

   @pytest.mark.gmosls
   @pytest.mark.parametrize("processed_flat", datasets, indirect=True)
   def test_regression_processed_flat(processed_flat, reference_ad):
      """
      Regression tests for reduce_FLAT recipe.

      Parameters
      ----------
      processed_flat : fixture
         returns an AstroData object containing a processed flat.
      reference_ad : fixture
         returns a function that loads the reference flat file.
      """
      ref_flat = reference_ad(processed_flat.filename)
      for ext, ext_ref in zip(processed_flat, ref_flat):
         np.testing.assert_allclose(ext.mask, ext_ref.mask)
         np.testing.assert_almost_equal(
            ext.data, ext_ref.data, decimal=3)
