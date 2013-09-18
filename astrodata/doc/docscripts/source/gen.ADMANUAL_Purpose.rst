
Abbreviations Table
-------------------
+ HDU: Header-Data Unit
+ MEF: Multi-Extension FITS
+ PHU: Primary Header Unit
+ URL: Uniform Resource Locator

Intended Audience
-----------------

This document is intended for both new and experienced developers using
``astrodata``:


#. Users of the ``astrodata`` package, in conjunction with the
   ``astrodata_Gemini`` configuration package.
#. Developers creating new configuration packages (types,
   descriptors, and transformations), e.g. instrument developers.
#. Potential developers needing to understand the work involved prior
   to development (e.g. for costing).
#. Those trying to understand both what the system currently does,
   it's design philosophy, and where the package can or is expected to
   evolve.



Document Structure
------------------

This document is meant as an introductory programmer reference for Gemini
Observatory's Python-based data processing package, ``astrodata``. It is
intended to serve both as an introductory reference for the actual
function interfaces of two primary classes in the astrodata package,
as well as a tool for new users to understand the general
characteristics of the package. To this end this document contains
three related but somewhat distinct sections:


+ Two chapters presenting the API reference
  manuals for the AstroData and ReductionContext classes, respectively.
+ A chapter on Creating an AstroData configuration Package, written as
  a hands-on startup guide.
+ A chapter on the Concepts in the AstroData Infrastructure.


The ``AstroData`` class is a dataset abstraction for MEF files, while the
``ReductionContext`` is the interface for transformation primitives to
communicate with the reduction system (eg. access files in the
pipeline, parameter information, execution context, and so on
including all communication with the system.)

The ``astrodata`` package includes only the infrastructure code, but is
generally shipped with the ``astrodata_Gemini`` configuration package
which contains all information and code regarding Gemini data types
and type-specific transformations, and with the ``astrodata_FITS`` configuration
package that contains standard FITS definitions.

The term "astrodata" in this document can refer to three somewhat
distinct aspects of the system. There is ``AstroData`` the class, which
is distinguishable in print by the camel caps capitalization and is
the core software element of the system. There is ``astrodata`` the
importable python package, which from the user's point of view imports
the configurations which are available in the environment, but which
strictly speaking contains only the infrustructural code. And there is
simply "Astrodata" a loose term for the whole package, including the
configuration package and support library.

