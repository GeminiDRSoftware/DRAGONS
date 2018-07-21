.. include howto

.. _intro:

************
Introduction
************
The DRAGONS Recipe System is Gemini Observatory's data processing software platform.
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
the ``reduce`` command line, and the programmatic interface on the ``Reduce``
class. The ``reduce`` command and programmatic access to the ``Reduce`` class
are the principle ways DRAGONS users can employ the Recipe System to process
and reduce their data.

The ``reduce`` application lets users invoke the Gemini Recipe System from 
the command line. As this document describes, the ``reduce`` command supports
a wealth of options that allow users to select and "tune" complex data processing
steps directly for one or more astronomical datasets.

Or not. Without any command line options, ``reduce`` will almost certainly
do the roughly correct thing by using well-tested default parameters for automated
processing. This is possible because an astronomical dataset (Gemini or otherwise)
encapsulated by ``AstroData`` exposes dataset properties and populates a *tags*
attribute on an instance of ``AstroData``. This *tags* property is a set of data
classifications describing the dataset, which is used by the Recipe System
to select the appropriate processing.

.. Defining the lexicon and coding the related actions allows ``AstroData`` and
the Recipe System infrastructure to naturally bring the following benefits:

.. • Instrument-agnostic programming
.. • Rapid development through isolation of dataset-specific heuristics
.. • A science-oriented data processing "recipe" system
.. • Automated and dynamic data processing
.. • Automatic propagation of associated data and metadata
.. • History, provenance, repeatability
.. • Support for building smart pipelines

As a quick example of how the Recipe System and ``AstroData`` work together, 
a typical ``reduce`` command can look deceptively simple, Without knowing the content
of the FITS file, you can simply run ``reduce`` on the data and the Recipe System
`mappers` automatically select the default recipe based upon the
data classifications presented by the dataset and ``AstroData``. Furthermore,
these data classifications have also been used to internally determine the most
applicable class of primitives from the set of defined instrument packages
(``targets``)::

 $ reduce S20161025S0111.fits
 			--- reduce, v2.0 (beta) ---
 All submitted files appear valid
 ===============================================================================
 RECIPE: reduce
 ===============================================================================
  PRIMITIVE: prepare
  ------------------
  ...

There are three critical parameters the Recipe System needs to map a dataset to
a primitive class and a recipe:

 * mode
 * recipe name
 * tags (or `tagset`)

Recipes and tags have already been mentioned, but ``mode`` is one other
parameter required to perform recipe selection.  The ``mode`` defines the
type of reduction one wants to perform: science quality, quick reduction
for quality assessment, etc.  Each ``mode`` defines its own recipe library.

For example, instrument packages for most Gemini instruments are provided under
the `gemindr` package, each of which contain a ``recipes`` directory that, in 
turn, contains a `qa` directory and a `sq` directory. These `mode` directories 
provide all recipes that pertain to data processing classified as Quality 
Assessment (``qa``) or Science Quality (``sq``).  Without indicating otherwise,
the default mode in the DRAGONS Recipe System is ``sq``. Users can request
``qa`` mode recipes by simply specifying the ``--qa`` switch on the command
line.

The DRAGONS Recipe System requires no naming convention on recipe
libraries or primitive filenames; the system is name-agnostic: naming of recipe
libraries, recipe functions, and primitive modules and classes is arbitrary. 

.. With no arguments passed on the command line, what has happened in the example
above? What has happened is that the Recipe System has fallen back to defaults
for a recipe name and a mode, which, in the current (beta) release, results
in the recipe, ``reduce``, and a mode of `sq`.

.. As indicated, a recipe is just a function that recieves a primitive instance
paired with the data, and which specifies that the following primitive functions 
are called on the data.

This is what a recipe looks like?
::

 def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image.

    Parameters
    ----------
    p : <primitives object>
        A primitive instance, usually returned by the PrimitiveMapper in
	a pipeline context, or any instantiated primitive class.
    """

    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.mosaicDetectors()
    p.makeFringe()
    p.fringeCorrect()
    p.alignAndStack()
    p.writeOutputs()
    return

As the reader can see, a recipe is essentially a sequence of primitives that
the data will be run through.  The primitives used at the recipe level
normally represent a clear, meaningful step in the reduction.  The recipe
is readable and does not require the scientific to know about Python coding
to figure out what will happen to the data.

.. The point here is not to overwhelm readers with a stack of primitive names, but
to present both the default pipeline processing that the above simple ``reduce`` 
command invokes and to demonstrate how much the ``reduce`` interface abstracts 
away the complexity of the processing that is engaged with the simplicity of 
commands.

There is much more to say about the topic of modes and recipe libraries, 
presented in depth in the :ref:`DRAGONS Recipe System Programmer’s Manual <refdocs>`.

Definitions
===========

Mode
----
The ``mode`` defines the type of reduction one wants to perform:
science quality, quick reduction for quality assessment, etc.
Each ``mode`` defines its own set of recipe libraries.
The mode is switched through command line flags.  Without a flag, the default
mode is science quality (``sq``).  Currently implemented are ``--qa`` and
``--ql``, indicating
that the Recipe System should map data to the Quality Assessment (``qa``)
recipes or to the Quick Look (``ql``) recipes.  ``qa`` recipes are used at
the observatory, at night, to evaluate the sky condition.  Those recipes
assume that the data come in one a time and calculates various metrics needed
for operations.  The ``ql`` recipes, which still have to be written, will
be used for quick reduction of target-of-opportunity and follow-up observation
with the goal of providing a reduced product good enough to assess the
scientific worth of the target and of the observations quickly.


Recipe
------
A recipe is a sequence of instructions specific to an instrument, type of
data, and recipe system mode.  It a Python function that calls a sequence
of ``primitives`` each designed to do one specific transformation.
A Recipe System mapper can select the recipe automatically.  Another mapper
selects the primitive set (collected through a ``primitive class``) the recipe
can use and passes it to it.


Recipe Library
--------------
A recipe library is a collection of recipes that applies to a specific
type of data.  The astrodata tags are used to match a recipe library to
a dataset.  The recipe library is implemented as Python module.  There can
be many recipes but only one is set as the default (though the user can
call any recipe within the library.)


Primitive
---------
A primitive is a defined function (method) on a primitive class. Under DRAGONS,
a primitive function is generally contrived to be a "science-oriented"
data processing step; for example, "bias correction." This is purely convention.
The Recipe System has no requirement that this be true.

Primitive Class
---------------
As defined under the DRAGONS package, ``geminidr``, primitive classes 
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

The DRAGONS packge must be installed and available, both at the command 
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
    DRAGONS/recipe_system/doc/rs_ProgManual/.

.. _related:

Related Documents
-----------------

  - `Astrodata cheat sheet`, Doc. ID: PIPE-USER-105_AstrodataCheatSheet,
    Cardenas, R., Gemini Observatory, 2017, astrodata/doc/ad_CheatSheet.

  - `Astrodata User’s Manual`, Doc. ID:  PIPE-USER-106_AstrodataUserManual,
    Labrie, K., Gemini Observatory, 2017, astrodata/doc/ad_UserManual/.

