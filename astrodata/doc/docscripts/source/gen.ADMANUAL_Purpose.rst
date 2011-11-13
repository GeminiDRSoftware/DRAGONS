


Document Purpose
----------------

This document is meant as an introductory user reference for Gemini
Observatory's python-based data processing package, "astrodata". It is
intended to serve both as an introductory reference for the actual
function interfaces of two primary classes in the astrodata package,
as well as a tool for new users to understand the general
characteristics of the package. To this end this document contains
three related but somewhat distinct sections:


+ The first chapters are API reference manuals for the AstroData and
  ReductionContext classes respectively.
+ An Appendix on Creating an AstroData configuration Package, written
  as a hands on startup-guide.
+ An Appendix on the Concepts in the AstroData Infrastructure


The AstroData class is a dataset abstraction for MEF files, while the
ReductionContext is the interface for transformation primitives to
communicate with the reduction system (i.e. access files in the
pipeline, parameter information, execution context, and so on
including all communication with the system.)

The astrodata package includes only the infrastructure code, but is
generally shipped with the astrodata_Gemini configuration package,
which contains all information and code regarding Gemini data types
and type-specific transformations. The astrodata package also ships
with an auxillary package of useful functions in the form of the
"gempy" package.

The term "astrodata" in this document can refer to three somewhat
distinct aspects of the system. There is "AstroData" the class, which
is distinguishable in print by the camel caps capitalization and is
the core software element of the system. There is "astrodata" the
importable python package, which from the user's point of view imports
the configurations which are available in the environment, but which
strictly speaking is only the infrustructural code. And there is
simply "Astrodata" a loose term for the whole package, including the
configuration package and support library.


Intended Audience
-----------------

This document is intended for both new and experience developers using
astrodata:

1. users of the astrodata package in conjunction with the
"astrodata_Gemini" configuration package 1. developers creating new
configuration information (types, descriptors, and transformations),
e.g. instrument developers 1. potential developers needing to
understand the work involved prior to development (e.g. for making
proposals) 1. those trying to understand both what the system
currently does, it's design philosophy, and where the package can or
is expected to evolve

