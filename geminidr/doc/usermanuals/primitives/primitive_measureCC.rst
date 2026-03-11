.. primitive_measureCC.rst

.. _primitive_measureCC:

*********
measureCC
*********
This primitive determines the extinction effects of telluric cloud cover for
an image dataset by comparing known and observed magnitudes of catalogued
objects in the observed field, with reference to magnitude zeropoints.

For datasets with multiple extensions, the normalization can either be applied
to the dataset as a whole, or on a per-extension basis.

For QA purposes, the primitive typically reports back the cloud cover (CC) level,
the measured zeropoint and associated error value, and how these compare to
originally-specified degree of acceptable cloud cover for the observation.

Implementations
***************

* :ref:`primitive_measureCC_gemini.qa`

.. _primitive_measureCC_gemini.qa:

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

.. include:: generated_doc/geminidr.gemini.primitives_qa.QA.measureCC_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.gemini.primitives_qa.QA.measureCC_param.rst

Algorithm
---------
This primitive determines the zeropoint by examining sources in
the object catalog for which a reference catalog magnitude has
been determined

It will also compare the measured zeropoint against the nominal
zeropoint for the instrument, and the nominal atmospheric extinction
as a function of airmass, to compute the estimated cloud attenuation.

This is intended for use with SExtractor-style source detection.
It relies on the prior addition of a reference catalog and source
cross-matching to populate the reference magnitude column of the object
catalog.

The reference magnitudes are taken directly from the reference
catalog. The measured magnitudes are taken directly from the object
detection catalog.

The correction for atmospheric extinction is made at the point at which
the zeropoint is calculated:

actual magnitude = zeropoint + instrumental magnitude + extinction correction

where in this case, the actual magnitude is the reference catalog magnitude,
the instrumental magntiude is obtained from from the object catalog, and
the nominal extinction value is used because a measured one s not available
at this point. In other words, the zeropoint is actually computed as::

zeropoint = refference magnitude - magnitude - nominal extinction correction

The zeropoint can then be treated as::

zeropoint = nominal photometric zeropoint - cloud extinction

in order to estimate the cloud extinction.

For datasets with multiple extensions, the mewasurement can either be applied
to the dataset as a whole, or on a per-extension basis by setting the
optional Boolean separate_ext parameter to True.

Issues and Limitations
----------------------
A reference catalog that includes positional and magnitude information must
already be avaialble.
