.. primitive_darkCorrect.rst

.. _primitive_darkCorrect:

***********
darkCorrect
***********
This primitive applies a dark correction to a set of one or more observed frames. The
image values of the specified dark frame(s) are subtracted from the observed frame(s),
in order to remove the unwanted dark current. If no dark is provided then the calibration
database is queried.

Implementations
***************

* :ref:`primitive_darkCorrect_core.preprocess`

.. _primitive_darkCorrect_core.preprocess:

Generic Implementation - core.primitive_preprocess module
===================================================
.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Top description from docstring
..      Inputs and Outputs section of the docstring
..      Parameters section of the docstring
..
..    The "Inputs and Outputs" section and the "Parameters" section in the
..    docstring must be underlined with "---" the length of the title for
..    compatibility with this document.  (Actually, this document was adapted
..    to use "---" as the section indicators at this level to match what we
..    already use in the docstrings.)

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.darkCorrect_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.darkCorrect_param.rst

Algorithm
---------
This primitive subtracts the input dark frames from the specified input observation
frames. The variance and data quality information will be updated accordingly, if
they exist. If no dark is provided then the calibration database is queried.


Issues and Limitations
----------------------
The inputs should have matching binning, shapes and units.
