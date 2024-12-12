.. glossary.rst

.. _glossary:

Glossary
========

.. glossary::

  astrodata

      Package distributed with the DRAGONS meta-package. ``astrodata`` is used
      to open datasets and provide an uniform interface to the data and the
      metadata (eg. headers) regardless of whether the file on disk is a FITS
      file or some other format, whether it is a GMOS file or NIRI file.  The
      Recipe System relies critically on ``astrodata``.

  AstroData

      Not to be confused with ``astrodata``, this is the base class for
      instrument-specific AstroData classes, and the one most users and
      developers will interact with at a programmatic level.

  descriptor
      A descriptor is a high-level access to essential dataset metadata
      (eg. headers) through a uniform, instrument-independent interface.
      E.g., ``ad.gain()``.  A descriptor is a method on an ``AstroData``
      instance.

  DRAGONS
      Data Reduction for Astronomy from Gemini Observatory North and South.

      A suite of packages comprising ``astrodata``, ``gemini_instruments``, the
      ``recipe_system``, ``geminidr``, and ``gempy``, which together provide
      the full functionality needed to run recipe pipelines on observational
      datasets. DRAGONS can be referred to as a framework.

  gempy
      A DRAGONS package comprising various functional utilities, some generic,
      some Gemini-specific.

  primitive
      A function defined within a data reduction instrument package that
      performs actual work on a dataset. Primitives observe controlled
      interfaces in support of re-use of primitives and recipes for different
      types of data, when possible. For example, all primitives called
      ``flatCorrect``  must apply the flat field correction appropriate for
      the data, and must have the same set of input parameters. This is a
      Gemini Coding Standard; it is not enforced by the Recipe System.

  recipe
      A function defined in a recipe library (module) which defines a sequence
      of calls to primitives. A recipe is a simple python function that receives
      an instance of the appropriate primitive class (primitive set) and
      executes the primitive sequence defined in the recipe function.  Users
      can pass recipe names directly to ``reduce.``

  Recipe System
      The DRAGONS framework that automates the selection and execution of
      recipes and primitives. The Recipe System defines a set of classes that
      uses attributes on an astrodata instance to locate recipes and primitives
      appropriate to the dataset.

  reduce
      The command line interface to the Recipe System.

  tags [or tagset]
      Represents a data classification. When loaded with ``AstroData``, a
      dataset will be classified with a number of tags that describe both the
      data and its processing state.  The tags are defined in *astrodata
      packages*, eg. the Gemini package is ``gemini_instruments``.
