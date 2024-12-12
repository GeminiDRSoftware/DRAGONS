.. intro.rst

.. _intro:

************
Introduction
************

Overview
========

The DRAGONS Recipe System is Gemini Observatory's data processing
automation platform. The Recipe System is designed to accommodate both
stepwise, interactive data processing, and automated data reduction pipelines.

The Recipe System inspect the inputs and automatically associates the recipes
and primitives most appropriate for those inputs.  A primitive is a step in
a reduction, for example ``biasCorrect``.  A recipe is a sequence of
primitives.  For the Gemini instruments, the collections of primitives and
recipes are found in the ``geminidr`` package. It is possible to specify
a different data reduction package.

The Recipe System relies on the Astrodata facility (``astrodata`` package) to
identify the input data and match them to the recipes and primitives.  The
Astrodata *tags* are the keys to the mapping.  For the Gemini instruments,
the Astrodata configurations are found in the ``gemini_instruments`` package.
Again, it is possible to specify a different Astrodata configuration package.

The ``reduce`` command and programmatic access to the ``Reduce`` class are the
principle ways DRAGONS users can employ the Recipe System to process and reduce
their data.   This document discusses a variety of examples of the ``reduce``
command line and the programmatic interface on the ``Reduce`` class.

The ``reduce`` command, and its programmatic interface, support options that
allow users to select and "tune" input parameters data processing steps.
Without any command line options or adjustment of the ``Reduce`` class
option attributes, the reduction uses default recipes and default input
parameters to the primitives.   In the ``geminidr`` package, which support
the Gemini instruments, the default recipes and primitive parametres have been
optimized to give good results in most cases.

A typical ``reduce`` command can look deceptively simple. Without knowing the
content of the data file, you can simply run ``reduce`` on the data and the
Recipe System automatically selects the best recipe and primitives based upon
the data classifications. For example, a call like this one can be all that
is needed::

 $ reduce S20161025S0111.fits
 			--- reduce, v2.0 (beta) ---
 All submitted files appear valid
 ===============================================================================
 RECIPE: reduce
 ===============================================================================
  PRIMITIVE: prepare
  ------------------
  ...
  ...


.. _refdocs:

Further Information
===================

Details and information on developing for the Recipe System, and about the
``astrodata`` package are available in companion manuals. We invite the reader
interested in those topics to refer to the topical documentation.

  - |RSProg|
  - |astrodatauser|
  - |astrodataprog|
