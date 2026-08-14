.. primitive_flatCorrect.rst

.. _primitive_flatCorrect:

***********
flatCorrect
***********
This primitive applies a flat-field correction to a set of one or more observed frames. The
image values of the specified observed frame(s) are divided through by the flat-field frame
values in order to generate flat-fielded data.

Implementations
***************

* :ref:`primitive_flatCorrect_core.preprocess`

.. _primitive_flatCorrect_core.preprocess:

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

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.flatCorrect_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.flatCorrect_param.rst

Algorithm
---------
This primitive divides the specified input observation frame by a normalized
flat field. The variance and data quality mask will be updated accordingly, if they
exist. If no flat-field frames are provided then the calibration database is
queried.

The algorithm verifies whether the flat-field includes a QE correction.  If
it does, a QE correction marker in the headers will be added to the output to
notify downstream primitives that the data has been QE corrected.

Issues and Limitations
----------------------
The inputs must have matching binning, shapes and units, as well as the same
number of extensions. The flat-field frame must not contain any zero values,
as this will result in a division by zero error.
