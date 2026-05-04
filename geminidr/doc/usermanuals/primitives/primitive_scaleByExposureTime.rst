.. primitive_scaleByExposureTime.rst

.. _primitive_scaleByExposureTime:

*******************
scaleByExposureTime
*******************
This primitive scales provided input images to have the same effective exposure
time. By default, the images will be scaled to match the exposure time of the first
image in the input list. Alternatively, the scaling parameter to be used can be
specified using the time parameter, 

Implementations
***************

* :ref:`primitive_scaleByExposureTime_core.preprocess`

.. _primitive_scaleByExposureTime_core.preprocess:

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

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.scaleByExposureTime_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_preprocess.Preprocess.scaleByExposureTime_param.rst

Algorithm
---------
This primitive scales provided input images to have the same effective exposure
time. The images will be scaled to match the exposure time of the first
image in the input list, or by a scaling parameter specified using the
time parameter, 

Issues and Limitations
----------------------
None.
