.. primitive_recordPixelStats.rst

.. _primitive_recordPixelStats:

****************
recordPixelStats
****************
This primitive enables addition of some basic statistics to the header of a dataset.
Typically, this might be the FITS header of the specified input dataset(s).
By default, the following items are added:

* PIXMEAN - the arithmetic mean of the pixel values;
* PIXSTDV - the standard deviation of the pixel values;
* PIXMED - the median of the pixel values.

The above are all calculated using the unmasked pixels only.

Implementations
***************

* :ref:`primitive_recordPixelStats_stats.stats`

.. _primitive_recordPixelStats_core.stats:

Generic Implementation - core.primitive_stats module
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

.. include:: generated_doc/geminidr.core.primitives_stats.Stats.recordPixelStats_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_stats.Stats.recordPixelStats_param.rst

Algorithm
---------
Standard NumPy mean, std and median functions are used.

Issues and Limitations
----------------------
The calculated parameters will only be added to the header if they are numerical
values. Non-numerical value, such as "NaN", etc. will not be added to the header.

