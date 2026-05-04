.. primitive_normalizeFlat.rst

.. _primitive_normalizeFlat:

*************
normalizeFlat
*************
This primitive normalizes a flat field dataset by an average of its data values.
By default, the mean is used for normalization, but the median can be used by setting
the scale parameter appropriately.

For datasets with multiple extensions, the normalization can either be applied
to the dataset as a whole, or on a per-extension basis.

Implementations
***************

* :ref:`primitive_normalizeFlat_core.preprocess`

.. _primitive_normalizeFlat_core.preprocess:

Generic Implementation - core.primitive_preprocess module
=========================================================
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

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.normalizeFlat_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.normalizeFlat_param.rst

Algorithm
---------
This primitive divides the specified input flat field observation by an average
of its data values, either as a whole or on a per-extension basis by setting
the optional Boolean parameter separate_ext to True.

By default, the mean is used for normalization, but the median can be used by setting
the scale parameter appropriately.

Issues and Limitations
----------------------
None.
