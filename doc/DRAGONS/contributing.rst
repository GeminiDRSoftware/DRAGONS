
Contributing
============

-  New code shall follow PEP-8. The **only** exceptions are the name of
   new Primitives and Parameters that may be semiCamelCase.

-  For each new file, the PyLint score must be greater 7.

-  New code must be tested following the `New Tests`_ guidelines.


New Tests
=========

- New tests should be written using
  `py.test <https://docs.pytest.org/en/latest/>`_.

- They can use testing modules included in packages, e.g.,
  ``numpy.testing``.

- Use `Fixtures <http://doc.pytest.org/en/latest/fixture.html>`_ for
  repetitive tasks.

- Use a `conftest.py file for local per-directory plugins <https://docs.pytest.org/en/2.7.3/plugins.html>`_.

- Tests must have ``test_`` prefix and have meaningful descriptive names.
  This is important because Jenkins will report more than a thousand of
  tests and they need to be readily understood. See the `Django Tutorial 05
  <https://docs.djangoproject.com/en/2.1/intro/tutorial05/>`_ for more examples.

::

    # Yes
    def test_can_perform_task(_):
        ...
        assert test_was_performed()

    # No
    def test_cpt(_):
        ...
        assert test_was_performed()

    # NEVER
    def test_1(_):
        ...
        assert test_was_performed()

- TestCase should be avoided since they don't work with fixtures and with
  parametrization.

- Tests must be added to the ``tests`` directory within the package
  directory.

- One test file should be created per module. If the test file gets too
  large, your module probably is also too large.


Tests using real data
=====================

- Tests using real data can be written.

- The data must be already public and available on the archive.

- All the data must be stored within a directory (sub-directories allowed).

- Every test file that uses real data should have the following fixtures

::

    @pytest.fixture
    def path_to_inputs():

        try:
            path = os.environ['DRAGONS_TEST_INPUTS']
        except KeyError:
            pytest.skip(
                "Could not find environment variable: $DRAGONS_TEST_INPUTS")

        if not os.path.exists(path):
            pytest.skip(
                "Could not access path stored in $DRAGONS_TEST_INPUTS: "
                "{}".format(path)
            )

        return path

    @pytest.fixture
    def output_test_path():

    try:
        path = os.environ['DRAGONS_TEST_OUT_PATH']
    except KeyError:
        warning.warn(
            "Could not find environment variable: $DRAGONS_TEST_OUT_PATH\n"
            "Using $DRAGONS_TEST_OUT_PATH="$(cwd)"
        )
        path = "."

    if not os.path.exists(path):
        pytest.skip(
            "Could not access path stored in $DRAGONS_TEST_OUT_PATH: "
            "{}".format(path)
        )

    return path

- The fixtures above can be added to the ``conftest.py`` instead if there
  are many test files.

- The first fixture above looks for the files that are cached inside a directory
  stored in the ``DRAGONS_TEST_INPUTS`` environment variable (case sensitive).

- The second fixture above checks if the ``DRAGONS_TEST_OUT_PATH`` environment
  variable exists. If so, this path will be used to store data produced during
  the tests. If not, it will print a warning message and the output path is the
  current working directory.

- Every file that will be cached for tests on Jenkins must be added to the
  ``.jenkins/test_files.txt``.

- ``.jenkins/test_files.txt`` allows comments for lines starting with ``#``.

- The file tree inside the ``.jenkins/test_files.txt`` reflects how the data
  will be organized within ``$TEST_PATH``.

- Existing files will be skipped.

- Cached files are kept between Jenkins builds.

- The ``.jenkins/download_test_data.py`` script can be used to download
  the data listed in the ``.jenkins/test_files.txt`` file and stored inside
  the ``$DRAGONS_TEST_INPUTS``. This script must be called from the DRAGONS
  root folder to work properly. The data is downloaded directly from the archive
  using ``curl``.
