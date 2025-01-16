.. definitions.rst

.. role:: bolditalic

.. _definition:

***********
Definitions
***********

When a reduction is launched with ``reduce`` (command line) or ``Reduce``
(Python class), the Recipe System will identify the nature of the inputs
using the :bolditalic:`AstroData tags`, and then start searching for the most
appropriate, or the requested, :bolditalic:`recipe` and
:bolditalic:`primitives`.

The Recipe System will search the active data reduction package (``geminidr``
or as specified by the ``--drpkg`` option) for :bolditalic:`recipe libraries`
and :bolditalic:`primitive sets` matching the inputs.  The
:bolditalic:`recipe library` search is limited in scope by the
:bolditalic:`mode` option.

Once everything has been found, the default or specified :bolditalic:`recipe`
from the selected :bolditalic:`recipe library` is given the
:bolditalic:`primitive set` as input.  The :bolditalic:`recipe` is run and
the sequence of :bolditalic:`primitive` calls is executed.

Below, we discuss each of the terms in :bolditalic:`bold italics` from the
execution summary above: "AstroData tags", "mode", "recipe", "recipe library",
"primitive", "primitive set".


AstroData Tags
==============
The ``AstroData Tags`` are data identification tags.  When a file is opened
with ``AstroData``, the software loads the *AstroData configuration modules* and
attempts to identify the data.

The tags associated with the dataset are compared to tags included in
recipes and in primitive classes.  The best match wins the selection process.

For Gemini instruments, the AstroData configurations are found in the ``gemini_instruments`` package.  This is set as the default. Which
configuration package to use can be configured on the ``reduce`` command line
or in the ``Reduce`` class.

More information on AstroData tags can be found in the |astrodatauser|.

Mode
====
The ``mode`` defines the type of reduction one wants to perform:
science quality ("sq"), quick look reduction ("ql"), or quality assessment
("qa"). Each ``mode`` defines its own set of recipe libraries. The mode is
switched through command line flags or the ``Reduce`` class ``mode`` attribute.

If not specified, the default is science quality, "sq".  Currently, only
science quality, quick look, and quality assessment are supported.  Users
cannot select other modes.

Recipe libraries of the same name but assigned different mode are often very
different from each other since the products are expected to be different.

The quality assessment mode, "qa", is used mostly at the Observatory, at night
to measure sky condition metrics and provide a visual assessment of the data. It
does not require calibrations since we might not have all the calibrations needed
at the time that the data was obtained.

The quick look mode, "ql", is intended for quick, close to but not necessarily
science quality reduction. The objective as the name entails being to do a
quick and automatic reduction for quick scientific and technical evaluation
of the data. This mode does not require calibrations either, but both QA and QL
modes can use calibrations if they are found.

The science quality mode, "sq", the default mode, is to be used in most cases.
The recipes in "sq" mode contain all the steps required to fully reduce data
without cutting corners. Some steps can be lengthy, some steps might offer
an optional interactive interface for optimization. This mode requires all
the calibrations and will return an error in case some are not found.

It is important to notice that a calibration processed with a lower quality
mode cannot be used by a higher quality mode (sq > ql > qa). For example, a
quicklook calibration cannot be used for science reduction, but a science
quality calibration can be used for a quicklook reduction.


Recipe
======
A recipe is a sequence of data processing instructions.  Technically, it is a
Python function that calls a sequence of *primitives*,  each primitive
nominally designed to do one specific transformation or service request.

Below is what a recipe can look like. This recipe performs the standardization
and corrections needed to convert the raw input science images into a stacked
image. The argument, ``p``, to the ``reduce`` recipe is the primitive set;
the recipe can call any primitives from that set.

::

 def reduce(p):
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

The guiding principle when building a recipe is to keep it human readable and
scientifically oriented.


Recipe Library
==============
A recipe library is a collection of recipes that applies to a specific
type of data.  The AstroData tags are used to match a recipe library to
a dataset.  A recipe library is implemented as Python module.  There can
be many recipes but only one is set as the default. It is however possible
for the user to override the default and call any recipe within the library.


Primitive
=========
A primitive is a data reduction step involving a transformation of the data or
providing a service.  By convention, the primitives are named to convey the
scientific meaning of the transformation. For example ``biasCorrect`` will
remove the bias signal from the input data.

A primitive is always a member of a primitive set.  It is the primitive set
that gets matched to the data by the Recipe System, not the individual
primitives.

Technically, a primitive is a method of a primitive class.  A primitive
class gets associated with the input dataset by matching the AstroData tags.
Once associated, all the primitives in that class, locally defined or inherited,
are available to reduce that dataset.  We refer to that collection of
primitives as a "primitive set".


Primitive Set
=============
A primitive set is a collection of primitives that are applicable to the
input dataset.  The association of the primitive set to the data is done by
matching AstroData tags.  It is a primitive set that gets passed to the recipe.
The recipe can use any primitive within that set.

Technically, a primitive set is a class that can have inherited from other more
general classes.  In ``geminidr``, there is a large inheritance tree of
primitive classes from very generic to very specific.  For example, the
primitive set for GMOS images defines a few of its own primitives and inherits
many other primitives from other sets (classes) like the one for
generic CCD processing, the one related to photometry, the one that applies to
all Gemini data, etc.
