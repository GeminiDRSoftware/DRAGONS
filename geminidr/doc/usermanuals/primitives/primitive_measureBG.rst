.. primitive_measureBG.rst

.. _primitive_measureBG:

*********
measureBG
*********
This primitive measures the sky background level for an image dataset by sampling
the non-object pixels in each extension.

The count levels are then converted to a flux using the nominal Zeropoint
values n orderi to determine the actual background level.

If the remove_bias option is set to True, then the bias level (if present)
is subtracted prior to determination of the background level.

For datasets with multiple extensions, the normalization can either be applied
to the dataset as a whole, or on a per-extension basis.

For QA purposes, the primitive typically reports back the observing band used,
the measured background sky brightness and associated error value, and how
these compare to requested weather band for the observation.

Implementations
***************

* :ref:`primitive_measureBG_gemini.qa`

.. _primitive_measureBG_gemini.qa:

Generic Implementation - gemini.primitive_qa module
=================================================
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

.. include:: generated_doc/geminidr.gemini.primitives_qa.QA.measureBG_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.gemini.primitives_qa.QA.measureBG_param.rst

Algorithm
---------
This primitive measures the sky background level for an image dataset by sampling
the unlfagged non-object pixel data values, either as a whole or on a per-extension
basis by setting the optional Boolean parameter separate_ext to True.

The count levels are converted to a flux using the nominal (not measured)
Zeropoint values - with the aim of measuring the actual background level, not the
flux incident on the top of the cloud layer necessary to produce that flux level.

For datasets with multiple extensions, the normalization can either be applied
to the dataset as a whole, or on a per-extension basis by setting the
optional Boolean separate_ext parameter to True.

Issues and Limitations
----------------------
None.
