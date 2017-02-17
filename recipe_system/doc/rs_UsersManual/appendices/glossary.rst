.. glossary.rst

.. _glossary:

Glossary
========

.. glossary::

  adcc

      Automatated Data Communication Center. Provides  HTTP 
      service for moinitoring QA metrics produced during pipeline operations. 
      This is run externally to ``reduce.`` Users need not know about or invoke 
      the ``adcc`` for ``reduce`` operations.

  astrodata

      Part of the gemini_python package suite 
      that defines the dataset abstraction layer for the Recipe System.

  AstroData

      Not to be confused with ``astrodata``, this is the base class for
      instrument-specific AstroData classes, and the one most users and
      developers will interact with at a programmatic level.

  descriptor
      Represents a high-level metadata name. Descriptors allow 
      access to essential information about the data through a uniform, 
      instrument-agnostic interface to the FITS headers.

  gemini_python
      A suite of packages comprising ``astrodata``, ``gemini_instruments``, the
      ``recipe_system`` and ``gempy``, all of which provide the full functionality
      needed to run recipe pipelines on observational datasets.

  gempy
      A ``gemini_python`` package comprising gemini specific functional utilities.

  MEF
      Multiple Extension FITS, the standard data format not only for Gemini
      Observatory but many observatories.

  primitive
      A function defined within a "dr" instrument package that performs actual
      work on the passed dataset. Primitives observe tightly controlled interfaces
      in support of re-use of primitives and recipes for different types of data,
      when possible. For example, all primitives called ``flatCorrect`` must apply
      the flat field correction appropriate for the data’s current astrdata tag set,
      and must have the same set of input parameters. This is a Gemini Coding
      Standard, it is not enforced by the Recipe System.

  recipe
      Represents the sequence of transformations, which are defined as methods
      on a primitive class. A recipe is a simple python function recieves an
      instance of the the appropriate primitive class and calls the available
      methods that are to be done for a given recipe function. A **recipe** is the
      high-level pipeline definition. Users can pass recipe names directly to
      ``reduce.`` Essentially, a recipe is a pipeline.

  Recipe System
      The gemin_python framework that accommodates an arbitrary number of defined
      recipes and primitives classes. 

  reduce
      The command line interface to the Recipe System and associated
      recipes/pipelines.
  
  tags [or tagset]
      Represent a data classification. A dataset will be classified by a number
      of tags that describe both the data and its processing state. For example,
      a typical unprocessed GMOS image might have the following tagset::

       set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])

      As Recipe System targets, instrument packages define tagsets as sets of
      string literals used to describe and map the kind of observational data
      that the package, primitive, or library has been defined to accommodate
      and process. As an example::

       set(['GMOS', 'MOS', 'NODANDSHUFFLE')]
