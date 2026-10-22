.. primitive_measureCC.rst

.. _primitive_measureCC:

*********
measureCC
*********
This primitive determines the extinction effects of telluric cloud cover (CC) for
an image dataset by comparing known and observed magnitudes of catalogued
objects in the observed field, with reference to magnitude zero points.

For datasets with multiple extensions, the normalization can either be applied
to the dataset as a whole, or on a per-extension basis.

For QA purposes, the primitive typically reports back the cloud cover level,
the measured zero point and associated error value, and how these compare to
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
This primitive measures the telluric cloud cover (CC) for an observational
dataset.  The zero point is determined by examining sources in
the object catalog for which a reference catalog magnitude has
been previously determined. It also compares the measured zero point
with the nominal zero point for the instrument, and the nominal telluric
extinction as a function of airmass to compute cloud attenuation.

It relies on the prior existence of a reference catalog and prior source
cross-matching to establish the reference magnitudes of the object
catalog.

The reference magnitudes are taken directly from the reference
catalog. The measured magnitudes are taken directly from the object
detection catalog.

The correction for atmospheric extinction is made when the zero point is
calculated, under the assumption of the relation

actual magnitude = zero point + instrumental magnitude + extinction correction

where in this case, the actual magnitude is the reference catalog magnitude,
the instrumental magntiude is obtained from from the object catalog, and
the nominal extinction value is used because a measured one is not available
at this point. In other words, the zero point is actually computed as::

zero point = reference magnitude - magnitude - nominal extinction correction

The measured zero point can then be assumed to be as follows::

zero point = nominal photometric zero point - cloud extinction

in order to estimate the cloud extinction.

For datasets with multiple extensions, the measurement can either be applied
to the dataset as a whole, or on a per-extension basis by setting the
optional Boolean separate_ext parameter to True.

Issues and Limitations
----------------------
A reference catalog that includes positional and magnitude information must
already be avaialble.
