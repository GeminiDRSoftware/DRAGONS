.. intro

Introduction
============

The AstroData class abstracts datasets stored in MEF files
and provides uniform interfaces for working on datasets from different
instruments and modes.  Configuration packages are used to describe
the specific data characteristics, layout, and to store type-specific
implementations.

MEFs can be generalized as lists of header-data units (HDU), with key-value
pairs populating headers, and pixel values populating the data array.
AstroData interprets a MEF as a single complex entity.  The individual
"extensions" within the MEF are available using Python list ("[]") syntax;
they are wrapped in AstroData objects. 

AstroData uses ``pyfits`` for MEF I/O and ``numpy`` for pixel manipulations.

While the ``pyfits`` and ``numpy`` objects are available to the programmer, 
``AstroData`` provides analogous methods for most ``pyfits`` functionalities
which allows it to maintain the dataset  as a cohesive whole. The programmer
does however use the ``numpy.ndarrays`` directly for pixel manipulation.

In order to identify types of dataset and provide type-specific behavior,
``AstroData`` relies on configuration packages either in the ``PYTHONPATH``
environment variable or the ``Astrodata`` package environment variables,
``ADCONFIGPATH`` and ``RECIPEPATH``. A configuration package 
(eg. ``astrodata_Gemini``) contains definitions for all instruments and
modes. A configuration package contains type definitions, meta-data 
functions, information lookup tables, and any other code
or information needed to handle specific types of dataset.

This allows ``AstroData`` to manage access to the dataset for convenience
and consistency. For example, ``AstroData`` is able:

 - to allow reduction scripts to have easy access to dataset classification
   information in a consistent way across all instruments and modes;
 - to provide consistent interfaces for obtaining common meta-data across all
   instruments and modes;
 - to relate internal extensions, e.g. discriminate between science and 
   variance arrays and associate them properly;
 - to help propagate header-data units important to the given instrument mode,
   but unknown to general purpose transformations.

In general, the purpose of ``AstroData`` is to provide smart dataset-oriented
interfaces that adapt to dataset type. The primary interfaces are for file
handling, dataset-type checking, and managing meta-data, but ``AstroData`` 
also integrates other functionalities.
