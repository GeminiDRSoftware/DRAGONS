.. glossary.rst

.. _glossary:

Glossary
========

.. glossary::

  astrodata

      Part of the DRAGONS package that defines the abstraction
      layer for observational datasets. The astrodata abstraction and its
      associated grammar is used extensively by the Recipe System to effect
      correct processing.

  AstroData

      Not to be confused with ``astrodata``, this is the base class for
      instrument-specific AstroData classes, and the one most users and
      developers will interact with at a programmatic level.

  descriptor
      Is a method on an ``AstroData`` instance. A descriptor represents a
      high-level metadata name and provides access to essential metadata
      through a uniform, instrument-agnostic interface to the FITS headers.
      E.g., ``ad.gain()``

  DRAGONS
      A suite of packages comprising ``astrodata``, ``gemini_instruments``, the
      ``recipe_system`` and ``gempy``, all of which provide the full functionality
      needed to run recipe pipelines on observational datasets.

  gempy
      A ``DRAGONS`` package comprising gemini specific functional utilities.

  primitive
      A function defined within a data reduction instrument package (a "dr" package)
      that performs actual work on the passed dataset. Primitives observe controlled
      interfaces in support of re-use of primitives and recipes for different types
      of data, when possible. For example, all primitives called ``flatCorrect``
      must apply the flat field correction appropriate for the data’s current
      astrodata tag set, and must have the same set of input parameters. This is a
      Gemini Coding Standard; it is not enforced by the Recipe System.

  recipe
      A function defined in a recipe library (module), which defines a sequence
      of function calls. A recipe is a simple python function that recieves an
      instance of the appropriate primitive class and calls the available methods
      that are to be done for a given recipe function. A **recipe** is the
      high-level pipeline definition. Users can pass recipe names directly to
      ``reduce.`` Essentially, a recipe is a pipeline.

  Recipe System
      The gemin_python framework that accommodates defined recipes and primitives
      classes. The Recipe System defines a set of classes that exploit attributes
      on an astrodata instance of a dataset to locate recipes and primitives
      appropriate to that dataset.

  reduce
      The command line interface to the Recipe System and associated
      recipes/pipelines.

  tags [tagset]
      Represent a data classification. A dataset will be classified by a number
      of tags that describe both the data and its processing state. For example,
      a typical unprocessed GMOS image taken at Gemini-South would have the
      following tagset::

       set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])

      Instrument packages define *tagsets*, which are sets of string literals
      that describe and the kind of observational data that the package,
      primitive, or library has been defined to accommodate and process.
      As an example::

       set(['GMOS', 'MOS', 'NODANDSHUFFLE')]
