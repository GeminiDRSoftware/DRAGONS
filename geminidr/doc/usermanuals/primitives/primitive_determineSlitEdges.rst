.. primitive_determineSlitEdges.rst

.. _primitive_determineSlitEdges:

******************
determineSlitEdges
******************
This primitive finds edges in flats, for the purpose of being able to mask out
unilluminated regions of the detector. It works for both longslit and cross-
dispersed data.

Implementations
***************

* :ref:`primitive_determineSlitEdges_core.spect`

.. _primitive_determineSlitEdges_core.spect:

Generic Implementation - core.primitives_spect module
=================================================================
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

.. include:: generated_doc/geminidr.core.primitives_spect.Spect.determineSlitEdges_docstring.rst

.. generated with `utility_scripts/generate_primdoc.py`
..    contains:
..      Parameter defaults from pex.config system
..      showpars-like format

.. include:: generated_doc/geminidr.core.primitives_spect.Spect.determineSlitEdges_param.rst

Algorithm
---------
To find edges, the primitive takes the first derivative of flux across the array
in the spatial diretion, searches for peaks to find rising edges, then searches
in an inverted copy of the array to find corresponding falling edges. It then
searches for pairs of edges with the expected width and near the expected pixel
locations (as determined from a LUT), using that information to weed out false
positive edge detections.

The locations of the edges are then used to trace them in the flux-derivative
array. The models derived from tracing are stored in a SLITEDGE table in the
extension, with columns for the slit and edge number, then the various coefficients
of the Chebyshev polynomials used for the edge models.

Finally, a distortion model is created from the combined traced coordinates of
all the edges. This will be used in distortionCorrect_ to rectify the slit(s),
i.e., to straighten them so that the spectra line up vertically or horizontally.


.. _distortionCorrect: primitive_distortionCorrect.rst

Issues and Limitations
----------------------
The repeatability of the positioning of the various optical components in Gemini
NIR spectral instruments can vary over a wide enough range that the edges of
the illuminated regions can be up to several dozen pixels different between
different observations (especially for GNIRS longslit observations). For some
longslit data, this could make the difference between one or both edges being
visible, so the algorithm contains a lot of code to handle these sorts of
situatiions.

----
