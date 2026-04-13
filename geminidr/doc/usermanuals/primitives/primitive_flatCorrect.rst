.. primitive_flatCorrect.rst

.. _primitive_flatCorrect:

***********
flatCorrect
***********
This primitive applies a flat-field correction to a set of one or more observed frames. The
image values of the specified observed frame(s) are divided through by the flat-field frame
values in order to generate flat-fielded data. If no flat field frames are provided then the
calibration database is queried.

If the flat field has had a QE correction applied, this information is copied into the
header iof the resultant frames, to prevent the correction being applied twice.

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
This primitive divides the specified input observation frame by the flat-field frames. The
variance and data quality information will be updated accordingly, if they exist. If no flat-field
frames are provided then the calibration database is queried.

Issues and Limitations
----------------------
The inputs should have matching binning, shapes and units.
