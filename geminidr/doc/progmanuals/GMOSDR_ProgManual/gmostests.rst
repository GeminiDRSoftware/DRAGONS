.. tests.rst

.. _gmostests:

***************
GMOS Test Suite
***************

Running the Tests
-----------------

.. note:: pytest instructions, a note about travis ci if relevant, a note
   that internal Gemini programmers have access to Jenkins (maybe).

    New developers are encouraged to check their modifications using the existing
tests before pushing the code back to the repository. Most of these tests uses
real data.


.. code-block:: bash

    export DRAGONS_TEST_INPUT="/path/to/input/"
    export DRAGONS_TEST_OUTPUT="/path/to/output/"
    export DRAGONS_TEST_REF="/path/to/ref/data/


.. code-block:: bash

    $ pytest geminidr/gmos



Available Tests
---------------

.. note:: as a list, or in a table

Missing or Desirable Tests
--------------------------

.. note:: as a list, or in a table
