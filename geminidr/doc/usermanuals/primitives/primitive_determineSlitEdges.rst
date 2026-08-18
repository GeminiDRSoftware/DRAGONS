.. primitive_determineSlitEdges.rst

.. _primitive_determineSlitEdges:

******************
determineSlitEdges
******************
This primitive finds and defines the edges between illuminated and
unilluminated areas in an image.  Generally, a well-illuminated flat is used.

A ``SLITEDGE`` table is added to each extension of the output AstroData object.
The table contains the coefficients of Chebyshev polynomials that describe the
edges of the illuminated regions of the detector.

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
in the spatial direction, searches for peaks to find rising edges, then searches
in an inverted copy of the array to find corresponding falling edges. It then
searches for pairs of edges with a separation matching the expected length and
near the expected pixel locations (as determined from a lookup table), using
that information to weed out false-positive edge detections.

The locations of the edges are then used to trace them in the flux-derivative
array. The models derived from tracing are stored in a Table named SLITEDGE
attached to the astrodata extension the measurements came from.  The columns
store the Chebyshev polynomial coefficients for the edge models, along
with columns for the slit and edge number, and the spectral order number
(for cross-dispersed data).

Here is an example of the SLITEDGE table for a GNIRS cross-dispersed
observation::

    >>> ad[0].SLITEDGE
    <Table length=12>
            c0                 c1                  c2                  c3          slit  edge specorder
         float64            float64             float64             float64       int64 int64   int64
    ------------------ ------------------ ------------------- ------------------- ----- ----- ---------
    265.65327088042665    59.584125425405 -1.2615273841680636 0.17693772967177296     1     0         3
     312.0585880182872  59.78164850850823 -1.2279372909483302 0.15472878588199565     1     1         3
      377.643110671976 45.527575577347015  1.2402314782456658  0.2746766394211974     2     0         4
     424.3290517531847  45.67494999293848  1.2553808589249997  0.2655383418222036     2     1         4
     457.0549552664068  48.41853693035303  3.4102869792025605 0.42748216929181027     3     0         5
    503.92178851500796 48.565936075351686   3.435559696232884  0.4369465431245844     3     1         5
      530.824371992081  59.91813539356759   5.712412364607932  0.5704622175506133     4     0         6
     577.8812058697536  60.10169456368148   5.777736902098741  0.5803061532502484     4     1         6
      608.481465444706    77.839436406476    8.45724001818417  0.7836506621668832     5     0         7
     655.6113282956126  77.82542463869505   8.382119649963984  0.7145423147912386     5     1         7
     694.2281167131091  101.0431254373623  11.433687733196319  0.9038701432959827     6     0         8
     741.7992681689526 101.45710896895568   11.59780168168031  0.9461162101794346     6     1         8


For longslit data, once the edges have been measureed, a distortion model is
created from the combined traced coordinates of both edges. This will be used
in distortionCorrect_ to rectify the slit, i.e., to straighten them so that the
spectra line up vertically or horizontally.   Note that this is not done for
cross-dispersed data. For cross-dispersed data, the rectification models cannot
be constructed and added to the WCS objects until after the slits have been
cut into individual astrodata extension in a later primitive.

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
