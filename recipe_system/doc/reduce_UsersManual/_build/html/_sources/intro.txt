.. intro:
.. include discuss

Introduction
============

This document is version 1.0 of the ``reduce`` Users Manual. This manual will 
describe the usage of ``reduce`` as an application provided by the Gemini Observatory 
Astrodata package suite. ``reduce`` is an application that allows users to invoke the 
Gemini Recipe System to perform data processing and reduction on one or more 
astronomical datasets.

This document presents details on applying ``reduce`` to astronomical datasets, 
currently defined as multi-extension FITS (MEF) files, both through the application's 
command line interface and the application programming interface (API). Details and 
information about the ``astrodata`` package, the Recipe System, and/or the data 
processing involved in data reduction are beyond the scope of this document and 
will only be engaged when directly pertinent to the operations of ``reduce``.

Reference Documents
-------------------

  - *The Gemini Recipe System: a dynamic workflow for automated data reduction*, 
    K. Labrie *et al*, SPIE, 2010.
  - *Developing for Gemini’s extensible pipeline environment*, K. Labrie, 
    C. Allen, P. Hirst, ADASS, 2011
  - *Gemini's Recipe System; A publicly available instrument-agnostic pipeline 
    infrastructure*, K. Labrie et al, ADASS 2013.

Overview
--------

As an application, ``reduce`` provides interfaces to configure and launch the 
Gemini Recipe System, a framework for developing and running configurable data 
processing pipelines and which can accommodate processing pipelines for arbitrary 
dataset types. In conjunction with the development of ``astrodata``, Gemini 
Observatory has also developed the compatible ``astrodata_Gemini`` package, the 
code base currently providing abstraction of, and processing for, Gemini 
Observatory astronomical observations.

In Gemini Observatory's operational environment "on summit," ``reduce``, 
``astrodata``, and the ``astrodata_Gemini`` packages provide a currently defined, 
near-realtime, quality assurance pipeline, the so-called QAP. ``reduce`` is used 
to launch this pipeline on newly acquired data and provide image quality metrics 
to observers, who then assess the metrics and apply observational decisions on 
telescope operations.

Users unfamiliar with terms and concepts heretofore presented should consult 
documentation cited in the previous sections (working on the Recipe System User 
Manual).


Glossary
--------

  **adcc** -- Automatated Data Communication Center. Provides XML-RPC and HTTP 
  services for pipeline operations. Can be run externally to ``reduce.`` Users 
  need not know about or invoke the ``adcc`` for ``reduce`` operations. 
  ``reduce`` will launch an ``adcc`` instance if one is not available. See 
  Sec. :ref:`adcc` for further discussion on ``adcc``.

  **astrodata** (or Astrodata) -- part of the **gemini_python** package suite 
  that defines the dataset abstraction layer for the Recipe System.

  **AstroData** -- not to be confused with **astrodata**, this is the main class 
  of the ``astrodata`` package, and the one most users and developers will 
  interact with at a programmatic level.

  **AstroDataType** -- Represents a data classification. A dataset will be 
  classified by a number of types that describe both the data and its processing 
  state. The AstroDataTypes are hierarchical, from generic to specific.  For 
  example, a typical unprocessed GMOS image would have a set of types like

  'GMOS_S', 'GMOS_IMAGE', 'GEMINI', 'SIDEREAL', 'IMAGE', 'GMOS', 'GEMINI_SOUTH', 
  'GMOS_RAW', 'UNPREPARED', 'RAW' (see **types** below).

  **astrodata_Gemini** -- the **gemini_python** package that provides all 
  observatory specific definitions of data types, **recipes**, and associated 
  **primitives** for Gemini Observatory data.

  **astrodata_X** -- conceivably a data reduction package that could reduce 
  other observatory and telescope data. Under the Astrodata system, it is 
  entirely possible for the Recipe System to process HST or Keck data, given 
  the development of an associated package, astrodata_HST or astrodata_Keck. 
  Pipelines and processing functions are defined for the particulars of each 
  telescope and its various instruments.

  **Descriptor** -- Represents a high-level metadata name. Descriptors allow 
  access to essential information about the data through a uniform, 
  instrument-agnostic interface to the FITS headers.

  **gemini_python** -- A suite of packages comprising **astrodata**, 
  **astrodata_Gemini**, **astrodata_FITS**, and **gempy**, all of which provide 
  the full functionality needed to run **Recipe System**  pipelines on 
  observational datasets.

  **gempy** -- a **gemini_python** package comprising functional utilities to 
  the **astrodata_Gemini** package.

  **MEF** -- Multiple Extension FITS, the standard data format not only for 
  Gemini Observatory but many observatories.

  **primitive** -- A function defined within an **astrodata_[X]** package that 
  performs actual work on the passed dataset. Primitives observe tightly 
  controlled interfaces in support of re-use of primitives and recipes for 
  different types of data, when possible. For example, all primitives called 
  ``flatCorrect`` must apply the flat field correction appropriate for the data’s 
  current AstroDataType, and must have the same set of input parameters.  This
  is a Gemini Coding Standard, it is not enforced by the Recipe System.

  **recipe** -- Represents the sequence of transformations. A recipe is a 
  simple text file that enumerates the set and order of **primitives** that will 
  process the passed dataset. A **recipe** is the high-level pipeline definition. 
  Users can pass recipe names directly to reduce. Essentially, a recipe is a 
  pipeline.

  **Recipe System** -- The gemin_python framework that accommodates an arbitrary 
  number of defined recipes and the primitives 

  **reduce** -- The user/caller interface to the Recipe System and its associated 
  recipes/pipelines.

  **subrecipe** -- Shorter recipe called like a primitive by a recipe or another
  subrecipe.  The subrecipes are not part of the main recipe index, they are more
  akin in purpose to primitives than to recipes. 
  
  **type** or **typeset** --  Not to be confused with language primitive or 
  programmatic data types, these are data types defined within an 
  **astrodata_[X]** package used to describe the kind of observational data that 
  has been passed to the Recipe System., Eg., GMOS_IMAGE, NIRI. In this document, 
  these terms are synonymous with **AstroDataType** unless otherwise indicated.
