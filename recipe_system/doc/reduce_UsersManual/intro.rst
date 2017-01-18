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
Observatory has also developed the ``gemini_instruments`` and ``GeminiDR`` 
packages, the code base currently providing abstraction of, and processing for, 
Gemini Observatory astronomical observations.

In Gemini Observatory's operational environment "on summit," ``reduce``, 
``astrodata``, and the ``gemini_instruments`` packages provide a currently defined, 
near-realtime, quality assurance pipeline, the so-called QAP. ``reduce`` is used 
to launch this pipeline on newly acquired data and provide image quality metrics 
to observers, who then assess the metrics and apply observational decisions on 
telescope operations.

Users unfamiliar with terms and concepts heretofore presented should consult 
documentation cited in the previous sections (working on the Recipe System User 
Manual).


Glossary
--------

  **adcc** -- Automatated Data Communication Center. Provides  HTTP 
  service for moinitoring QA metrics produced during pipeline operations. 
  This is run externally to ``reduce.`` Users need not know about or invoke 
  the ``adcc`` for ``reduce`` operations.

  **astrodata** (or Astrodata) -- part of the **gemini_python** package suite 
  that defines the dataset abstraction layer for the Recipe System.

  **AstroData** -- not to be confused with **astrodata**, this is the main class 
  of the ``astrodata`` package, and the one most users and developers will 
  interact with at a programmatic level.

  **AstroData tags** -- Astrodata tags Represents a data classification. A dataset 
  will be classified by a number of types that describe both the data and its 
  processing state. For example, a typical unprocessed GMOS image would have a 
  set of tags like

  set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])
  (see **tags** below).

  **Descriptor** -- Represents a high-level metadata name. Descriptors allow 
  access to essential information about the data through a uniform, 
  instrument-agnostic interface to the FITS headers.

  **gemini_python** -- A suite of packages comprising **astrodata**, 
  **gemini_instruments**, the **recipe system** and **gempy**, all of which 
  provide the full functionality needed to run recipe  pipelines on 
  observational datasets.

  **gempy** -- a **gemini_python** package comprising gemini specific functional 
  utilities.

  **MEF** -- Multiple Extension FITS, the standard data format not only for 
  Gemini Observatory but many observatories.

  **primitive** -- A function defined within an **GeminiDR** package that 
  performs actual work on the passed dataset. Primitives observe tightly 
  controlled interfaces in support of re-use of primitives and recipes for 
  different types of data, when possible. For example, all primitives called 
  ``flatCorrect`` must apply the flat field correction appropriate for the data’s 
  current astrdata tag set, and must have the same set of input parameters.  This
  is a Gemini Coding Standard, it is not enforced by the Recipe System.

  **recipe** -- Represents the sequence of transformations, which are defined as
  methods on a primitive class. A recipe is a simple python function recieves an 
  instance of the the appropriate primitive class and calls the available methods 
  that are to be done for a given recipe function. A **recipe** is the high-level 
  pipeline definition. Users can pass recipe names directly to reduce. Essentially, 
  a recipe is a pipeline.

  **Recipe System** -- The gemin_python framework that accommodates an arbitrary 
  number of defined recipes and the primitives 

  **reduce** -- The command line interface to the Recipe System and its associated 
  recipes/pipelines.
  
  **tags** or **tag set** --  these are tags that characterise the dataset and 
  defined in a ``gemini_instruments`` instrument package used to describe the 
  kind of observational data that has been passed to the Recipe System., 
  Eg., a GMOS IMAGE; a NIRI IMAGE.
