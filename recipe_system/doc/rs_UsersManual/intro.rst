.. include howto

.. _intro:

************
Introduction
************
The Recipe System is Gemini Observatory's data processing software platform.
The Recipe System is designed as a framework that accommodates both stepwise,
interactive data processing and automated data reduction pipelines.

The Recipe System comprises a set of what are called `mapper` classes, which, 
as the name implies, "map" input datasets to discoverable classes of `primitives` 
and to the functions we call `recipes`. The Recipe System further comprises the 
class `Reduce` and the script `reduce`, which respectively support an application 
programming interface (API) and a command line interface. The Recipe System 
requires that data reduction `instrument packages` provide defined `primitive 
classes` and `recipe` functions. In this document, `instrument packages` may
be referred to as `targets` of the Recipe System.

This document presents information, discussion, and a wealth of examples of 
the ``reduce`` command line, and the programmatic interface on ``Reduce`` class. 
The ``reduce`` command and programmatic access to the ``Reduce`` class are the 
primary ways that users can use the Recipe System to process and reduce their data.

The ``reduce`` application allows users to invoke the Gemini Recipe System from 
the command line to perform complex data processing and reduction on one or more 
astronomical datasets with a minimal set of parameters when default processing is 
requested. This is possible because an astronomical dataset (Gemini or otherwise) 
encapsulated by ``AstroData`` exposes dataset properties and populates a *tags* 
attribute on an instance of the class to provide an interface to dataset 
information and also to present a FITS file as a cohesive set of the data. 
Defining the lexicon and coding the related actions allows ``AstroData`` and
the Recipe System infrastructure to naturally bring the following benefits:

 • Instrument-agnostic programming
 • Rapid development through isolation of dataset-specific heuristics
 • A science-oriented data processing "recipe" system
 • Automated and dynamic data processing
 • Automatic propagation of associated data and metadata
 • History, provenance, repeatability
 • Support for building smart pipelines

As a quick example of how the Recipe System and ``AstroData`` work together, 
a typical reduce command can look deceptively simple::

 $ reduce S20161025S0111.fits
 			--- reduce, v2.0 (beta) ---
 All submitted files appear valid
 ===============================================================================
 RECIPE: reduce_nostack
 ===============================================================================
 ...

Without knowing the content of the FITS file, you can simply run `reduce` on the 
data and the Recipe System `mappers` automatically select the default recipe, 
``reduce_nostack``, based upon the data classifications presented by the dataset 
and ``AstroData``. Furthermore, these data classifications have also been used 
to internally determine the most applicable class of primitives from the set of 
defined instrument packages (`targets`).

There are three critical parameters the Recipe System needs to map a dataset to
a primitive class and a recipe:

 * context
 * recipe name
 * tags (or `tagset`)

Recipes and tags have already been mentioned, but `context` is one other 
parameter required to perform recipe selection. The `context` is simply a 
label by which the recipe libraries are delineated and which are manifest 
in instrument packages as directories named with these same context labels.

For example, instrument packages for many Gemini instruments are provided under 
the `gemindr` package, each of which contain a ``recipes`` directory that, in 
turn, contains a `qa` directory and a `sq` directory. These `context` directories 
provide all recipes that pertain to data processing classified as Quality 
Assurance (``qa``) or Science Quality (``sq``). The **QAP**, the Quality 
Assurance Pipeline, is the ``reduce_nostack`` recipe under the ``qa`` context 
libraries. As users become familiar with running and using the Recipe System, 
they are free to add any new contexts as they like. The Recipe System is 
name-agnostic: contexts, recipe libraries (modules), and recipe function names 
are arbitrary.

With no arguments passed on the command line, what has happened in the example 
above? What has happened is that the Recipe System has fallen back to defaults
for a recipe name and a context, which, in the current (beta) release, results
in the recipe, ``reduce_nostack``, and a context of `qa`. These default
settings are equivalent to the command line::

 $ reduce -r reduce_nostack --context qa S20161025S0111.fits

As indicated, a recipe is just a function that recieves a primitive instance 
paired with the data, and which specifies that the following primitive functions 
are called on the data. And what does the ``reduce_nostack`` recipe look like? 
::

 def reduce_nostack(p):
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.addReferenceCatalog()
    p.determineAstrometricSolution()
    p.measureCC()
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.detectSources()
    p.measureIQ(display=True)
    p.measureBG()
    p.determineAstrometricSolution()
    p.measureCC()
    p.addToList(purpose='forStack')
    p.writeOutputs()
    return

The point here is not to overwhelm readers with a stack of primitive names, but 
to present both the default pipeline processing that the above simple ``reduce`` 
command invokes and to demonstrate how much the ``reduce`` interface abstracts 
away the complexity of the processing that is engaged with the simplicity of 
commands.

There is much more to say about the topic of contexts and recipe libraries, 
presented in depth in the :ref:`Recipe System Programmer’s Manual <refdocs>`.

Definitions
===========

Context
-------
A context is a label by which the recipe libraries are delineated and 
which are manifest in instrument packages as directories named with these 
same labels. These context names `should` indicate or hint at the purpose or 
quality of the recipes contained therein. For example, Quality Assurance recipes 
are found in the ``qa`` recipes directory, Science Qauality recipes, in an 
``sq`` recipes directory.

Recipe
------
A recipe is a python function defined for specific instruments and modes. A
recipe function recieves one parameter, an instance of a primitive class. 
This "primitive" class presents all available primitive methods on the 
instance recived by the recipe, which is then free to call any primitive 
function in any order. The acquisition of an applicable recipe and primitive
class is the primary operation provided by ``reduce``.

Recipe Library
--------------
A python module defined in an instrument package that comprises one or more 
defined *recipes*. A recipe library (module) will have one (1) attribute
defined as ``recipe_tags``, which is a set of tags indicating the kind of
data to which this recipe library applies.

Primitive
---------
A primitive is a defined method on a primitive class. A primitive function 
is generally contrived to be a "science-oriented" data processing step, for
example, "bias correction," though the Recipe System has no requirement
that this be true.

Primitive Class
---------------
As defined under the *gemini_python* package, ``geminidr``, primitive classes 
are a large set of hierarchical classes exhibiting inheritance from generic to
specific. Because they are real data, datasets will always have some 
instrument/mode specific set of *tags* that will allow the Recipe System to pick
instrument/mode specific primitive class. 

Further Information
===================
As this document details, ``reduce`` provides a number of options and command 
line switches that allow users to control the processing of their data.
This document will further describe usage of the ``Reduce`` class' API. A 
detailed presentation of the above components comprise Chapter 3, :ref:`howto`.

The *gemin_python* packge must be installed and available, both at the command 
line and as importable Python packages. The :ref:`next chapter <install>` 
takes readers through the install process.

Details and information about the ``astrodata`` package, the Recipe System, 
and/or the data processing involved in data reduction are beyond the scope of 
this document and will only be engaged when directly pertinent to the operations 
of the Recipe System. Users and developers wishing to see more information about 
how to use the programmtic interfaces of the Recipe System should consult the
documents described next section.

.. _refdocs:

Reference Documents
-------------------

  - `RecipeSystem v2.0 Design Note`, Doc. ID: PIPE-DESIGN-104_RS2.0DesignNote,
    Anderson, K.R., Gemini Observatory, 2017, DPSGdocuments/.

  - `Recipe System Programmer’s Manual`, Doc. ID: PIPE-USER-108_RSProgManual,
    Anderson, K.R., Gemini Observatory, 2017, 
    gemini_python/recipe_system/doc/rs_ProgManual/.

.. _related:

Related Documents
-----------------

  - `Astrodata cheat sheet`, Doc. ID: PIPE-USER-105_AstrodataCheatSheet,
    Cardenas, R., Gemini Observatory, 2017, astrodata/doc/ad_CheatSheet.

  - `Astrodata User’s Manual`, Doc. ID:  PIPE-USER-106_AstrodataUserManual,
    Labrie, K., Gemini Observatory, 2017, astrodata/doc/ad_UserManual/.


The Recipe System is Gemini's data processing software platform for end-users
reducing data on their computer. However, the Recipe System is also designed to 
form the heart of automated data reduction pipelines.
