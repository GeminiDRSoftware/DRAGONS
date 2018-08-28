.. interfaces.rst
.. include mappers
.. include overview

.. _iface:

Using the Mappers API
*********************
For practical applications, the Mapper base class provides no functionality, but
defines all attributes for instances built from subclasses of Mapper. Though not 
strictly an abstract class, the Mapper base class cannot be used on its own. 
Subclasses, such as PrimitiveMapper and RecipeMapper, should not override 
Mapper.__init__(), but implement their own mapping routines optimized to the 
"target" of the mapping. An example of a Mapper extension might be an 
implemention of a mapper to find "applicable" lookup tables in instrument 
packages.

The programmatic interfaces on the current mapper classes are straight forward.
One begins by passing a list of astrodata instances, and any ancilliary arguments,
to the "constructor" of either PrimitiveMapper or RecipeMapper. Below, we reiterate
the input arguments to a Mapper class and show the default values of parameters not
passed by the caller::

  __init__(self,
           adinputs,             <list> AstroData objects.
	   mode='sq',             <str> Defines the recipe libraries to search.
	   drpkg='geminidr',      <str> Defines the 'dr' package to map.
	   recipename='default',  <str> The recipe name.
           usercals=None,        <dict> User provided calibration files.
	   uparms=None,          <list> User parameters passed to primitives.
	   upload=None           <list> Send these things to fitsstore
           )

Once an instance of either a PrimitiveMapper or a RecipeMapper class is built, 
that instance has one (1) and only one public method, a method that invokes
the search algorithm for the instance.

It shall be noted here that the following discussion and examples are based on
using the default data reduction package, *geminidr*. The *geminidr* package
defines all recipes, modes, and primitive classes for several instruments of the
Observatory. With that in mind, the last :ref:`section of this chapter <drpkg>`
will detail steps required to build your own "drpkg", whether for testing purposes
or as a new, complete data reduction package. It will be the case that all
examples presented herein will be perfectly applicable to any correctly
implemented *drpkg*.

Selecting Primitives with PrimitiveMapper
=========================================

Primitive classes for Gemini Observatory instruments are defined in the *geminidr*
package under DRAGONS. This is specified as the default value on the ``drpkg``
keyword argument shown above. These primitive classes define methods to provide
essential data processing functionality. Primitive classes in *geminidr* are
structured hierarchically and employ multiple inheritance. (Hereafter, a Primitive
class may be referred to as a set of "primitives" or just "primitives", which are
just the defined or inherited methods on that class).

"Generic" primitive classes in the ``geminidr`` package are defined under
``geminidr.core`` (see :ref:`Figure 4.1, Primitive Class Hierarchy <prmcls>`.
These generic classes provide functions that work on all data produced by Gemini
Oberservatory. These classes are arranged logically, meaning primitive functions
for some general task are grouped together. For example, the stacking functions
are defined on the ``Stack`` class found in ``core.primitives_stack``.

There are five (5) defined primitive classes in `core` that are not strictly
generic but are what might be called "quasi-generic". That is, these classes
define methods for data of a certain general kind, like imaging or spectroscopy.
:ref:`Figure 4.2 <gmoscls>` illustrates these classes by breaking them out of
*core* to show what they are and where in the class structure they are used.

Generic classes are inherited by the primitive class, ``Gemini``. The ``Gemini``
class is then inherited by all instrument-specific primitive classes. These
instrument-specific classes are what might be called "concrete," because they
will provide a complete set of fully implement methods particular to the
instrument data being processed.

.. _prmcls:

.. figure:: images/primcls16.jpg

   Hierarchy of Primitive classes defined under `geminidr`

Because real data are produced by real instruments, the PrimitiveMapper will
usually be selecting primitive classes defined at the instrument-mode
level, i.e., one or more inheritance levels under an instrument primitive class.
That sounds like gobble but :ref:`Figure 4.1, Primitive Class Hierarchy <prmcls>`,
illustrates that this is simple. For example, an F2 image will be processed with
the "f2 image" primitive class, GNIRS image data, the "gnirs image" class, and so
on.

.. _gmoscls:

.. figure:: images/primcls_gmos3.jpg

   Hierarchy of Primitive classes inherited by GMOS

Recall that primitive classes are attributed with a *tagset* indicating the
particular kinds of data to which they are applicable. Indeed, as defined in the
*geminidr* package, only ``gemini`` and subclasses thereof have *tagset*
attributes that make them discoverable by the PrimitiveMapper. Which also
implies that any primitive classes defined in ``core`` are not discoverable by
the PrimitiveMapper. We shall examine the details of this statement in the next
section.

Mapping Data to Primitives
--------------------------

When the PrimitiveMapper receives input data, those data are passed as a
list of *astrodata* objects, one *astrodata* object per input dataset. All
astrodata objects have been classified with a number of what are called `tags`,
which are present on the *astrodata* instance as an attribute of the object.
For example, a typical unprocessed GMOS image:

>>> ad = astrodata.open('S20161025S0111.fits')
>>> ad.tags
set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])

The PrimitiveMapper uses these tags to search *geminidr* packages, first by 
immediately narrowing the search to the applicable instrument package. In this 
case, the instrument and package are ``gmos``. The Mapper classes have an
understanding of this, and set their own attribute on Mapper instances called,
``pkg``:

>>> from recipe_system.mappers.primitiveMapper import PrimitiveMapper
>>> pm = PrimitiveMapper([ad])
>>> pm.pkg
'gmos'

Once a PrimtiveMapper instance is created, the public method, 
``get_applicable_primitives()`` can be invoked and the search for the most 
appropriate primitive class begins. The search itself is focused on finding
class objects that define a ``tagset`` attribute on the class.

Let's see how primitive classes in the hierarchy are tagged, beginning with
``Gemini`` class::

  class Gemini( ...  ):
    tagset = set(["GEMINI"])

  class GMOS(Gemini, ... ):
      tagset = set(["GEMINI", "GMOS"])

  class GMOSImage(GMOS, ... ):
      tagset = set(["GEMINI", "GMOS", "IMAGE"])

The PrimitiveMapper gloms all primitive classes in the package, looking for a 
maximal subset of the *astrodata tags* in the tagset attribute of the primitive 
classes. Using our astrodata ``tags`` in the example above, we can see that 
``GMOSImage`` class provides a maximal matching tagset to the astrodata object's 
data classifications.

We proceed from the example above and have the PrimitiveMapper do its job:

>>> pset = pm.get_applicable_primitives()

Check that we have the primitives we expect:

>>> pset.__class__
<class 'geminidr.gmos.primitives_gmos_image.GMOSImage'>

Which is exactly correct. Once PrimitiveMapper has acquired the best "applicable"
primitive class, it instantiates the primitives object using the parameters 
passed. The returned ``pset`` is the *actual instance of the class* and is ready 
to be used.

The *tagset* is the only criterion used by the PrimitiveMapper to find the correct
primitive class. Readers may correctly infer from this that naming primitive
classes, and the modules containing them, is arbitrary; primitive classes and the
containing modules can be named at the discretion of the developer. Indeed, the
entire set of primitive classes could exist in a single file. For reasons too
obvious to enumerate here, such an "arrangement" is considered ill-advised.

.. _rselect:

Selecting Recipes with RecipeMapper
===================================

Recipes are pre-defined python functions that receive a single argument: an
instance of a primitive class. Unlike primitive classes, recipes are much
simpler; they are straight up functions with one argument. Recipe functions are
not classes and do not (cannot) inherit. The recipe simply defines the set and
order of primitive functions to be called on the data, references to which are
contained by the primitive instance. Essentially, a recipe is a pipeline.

Recipe functions are defined in python modules (which may be referred to as
recipe libraries, a collection of functions) that are placed in a *geminidr*
instrument package. Recipes are only defined for instruments and exist under
an instrument package in a ``recipes/`` directory like this::

  ../geminidr/f2/recipes
  ../geminidr/gmos/recipes
  ../geminidr/gnirs/recipes
  .. [etc. ]

Here is a (current) listing of instrument recipe directories under *geminidr*::

  geminidr/f2/recipes/:
      __init__.py
      qa/
      sq/

  geminidr/gmos/recipes/:
      __init__.py
      qa/
      sq/

  geminidr/gnirs/recipes/:
      __init__.py
      qa/
      sq/

  geminidr/gsaoi/recipes/:
      __init__.py
      qa/
      sq/

  geminidr/niri/recipes/:
      __init__.py
      qa/
      sq/

Readers will note the appearance of directories named ``qa`` and ``sq`` under
recipes. These directories indicate a separation of recipe types, named to
indicate the kinds of recipes contained therein. Any named directories defined
under ``recipes/`` are termed "modes."

.. _mode:

Mode
----
An instrument package *recipes* path is extended by names indicating a "mode."
As shown above, *geminidr* instrument packages define two modes under all
recipes directories: `qa` and `sq`. These indicate that recipes defined under
``recipes/qa`` provide Quality Assurance (*qa*) processing. Science Quality
(*sq*) recipes defined under ``recipes/sq`` provide science quality reduction
pipelines. Currently defined recipe library files will appear under one or all of
these mode directories.

Currenntly, mode values are hard limited to `qa`, `ql`, and `sq` modes for the
RecipeMapper. As a refresher, readers are encouraged to review the command line
options provided by *reduce*, where *mode* is discussed in detail in the document,
`Reduce and Recipe System User Manual`.

Discussion of instrument packages and their format are presented in some detail 
in the section of Chapter 2, :ref:`Instrument Packages <ipkg>`.

.. _d2r:

Mapping Data to Recipes
-----------------------

When the RecipeMapper receives input data, those data are passed as a
list of *astrodata* objects, one *astrodata* object per input dataset. All
astrodata objects have been classified with a set of `tags`, which are present
on the *astrodata* instance as an attribute of the object. For example, a
typical unprocessed GMOS image:

>>> ad = astrodata.open('S20161025S0111.fits')
>>> ad.tags
set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])

The RecipeMapper uses these tags to search *geminidr* packages, first by
immediately narrowing the search to the applicable instrument package and then
by using the ``mode`` parameter, further focusing the recipe search. In this
case, the instrument and package are ``gmos``. The Mapper classes have an
understanding of this, and set their own attribute on Mapper instances called,
``pkg``:

>>> from recipe_system.mappers.recipeMapper import RecipeMapper
>>> rm = RecipeMapper([ad])
>>> rm.pkg
'gmos'

You can also see the current mode, in this case, the 'default' setting on
the RecipeMapper instance:

>>> rm.mode
'sq'

Should you want to have the RecipeMapper search for *qa* recipes, simply set the
attribute:

>>> rm = RecipeMapper([ad])
>>> rm.mode
'sq'
>>> rm.mode = 'qa'

Once a RecipeMapper instance is created and attributes have been set as desired,
the public method, ``get_applicable_recipe()`` can be invoked and the search for
the most appropriate recipe begins. The search algorithm is concerned with finding
module objects that define a ``recipe_tags`` attribute on the module (library).
Each recipe library defines, or may define, multiple recipe functions, all of
which are applicable to the data classification described by the ``recipe_tags``
set.

Continuing the 'gmos' example, let's see how these recipe libraries are tagged::

  gmos/recipes/qa/recipes_BIAS.py:
  -------------------------------
  recipe_tags = set(['GMOS', 'CAL', 'BIAS'])

  gmos/recipes/qa/recipes_FLAT_IMAGE.py:
  -------------------------------
  recipe_tags = set(['GMOS', 'IMAGE', 'CAL', 'FLAT'])

  gmos/recipes/qa/recipes_IMAGE.py:
  -------------------------------
  recipe_tags = set(['GMOS', 'IMAGE'])

  gmos/recipes/qa/recipes_NS.py:
  -------------------------------
  recipe_tags = set(['GMOS', 'NODANDSHUFFLE'])


The RecipeMapper gloms all recipe libraries in the package, looking for a 
maximal subset of the *astrodata tags* in the ``recipe_tags`` attribute of the 
recipe library. Referring to the astrodata ``tags`` in the example above, simple
inspection reveals that the ``recipes_IMAGE`` library for GMOS provides a maximal 
matching *subset* of tags to the astrodata object's data classifications.

A Running Example
-----------------

The example that follows begins by first making an ``astrodata`` instance 
from an arbitrary FITS file, passing that alone to the RecipeMapper, and then 
calling the instance's public method, ``get_applicable_recipe()``.

>>> import astrodata
>>> import gemini_instruments
>>> ad = astrodata.open('S20161025S0111.fits')
>>> ad.tags
set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])
>>> adinputs = [ad]
>>> from recipe_system.mappers.recipeMapper import RecipeMapper
>>> rm = RecipeMapper(adinputs)
>>> recipe = rm.get_applicable_recipe()
>>> recipe.__name__ 
'reduce'

.. note:: Remember, `adinputs` must be a *list* of astrodata objects.
   
Set mode
^^^^^^^^

Let's say we are uncertain of which recipe mode we actually used. Simply
inspect the mapper object:

>>> >>> rm.mode
'sq'

But, it turns out that we would like to get the default 'qa' recipe, not the 
default 'sq' recipe. All we need to do is set the mode attribute on the 
RecipeMapper object and the recall the method:

>>> rm.mode = 'qa'
>>> recipefn = rm.get_applicable_recipe()
>>> recipefn.__name__
'reduce_nostack'

Which is the defined default recipe for the GMOS `qa` recipe mode.

As this returned recipe function name suggests, image stacking will not be done.
But perhaps we might want to use a recipe that does perform stacking. We simply
set the recipename attribute to be the desired recipe. [#]_

>>> rm.recipename = 'reduce'
>>> recipefn = rm.get_applicable_recipe()
>>> recipefn.__name__
'reduce'

There is more going on here than simply setting a string value to the
recipename attribute. The RecipeMapper is actually acquiring the named recipe
using the already set *mode* and the astrodata tagset. Calling the method a
second time relaunches the search algorithm, this time for the `qa` mode, 
imports the "applicable" `qa` recipe function and returns the function object 
to the caller.

Returning to the class initializer, we can get this same result by passing the 
relevant arguments directly to the RecipeMapper call.

>>> rm = RecipeMapper(adinputs, mode='qa', recipename='reduce')
>>> recipefn = rm.get_applicable_recipe()
>>> recipefn.__name__
'reduce'

Selecting External (User) Recipes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, let's say we have an external recipe function defined in a file named, 
``myrecipes.py`` in some arbitrary location and would like to use that recipe. 
While you know the file name, location, and the recipe function name, the 
RecipeMapper does the work of importing the file and returning the function 
object in one easy step.

While some users may have set their ``PYTHONPATH`` to include such arbitrary 
locations, which would allow the ``myrecipes`` module to be imported directly, 
most people will not have such paths in their ``PYTHONPATH``, and would not be 
able to directly import their recipe file without modifying their environment. 
Using the RecipeMapper lets users avoid this hassle because it handles import 
transparently.

E.g.,
  
>>> rm.recipename = '/path/to/myrecipes.myreduce'
>>> recipefn = rm.get_applicable_recipe()
>>> recipefn.__name__
'myreduce'

We may obtain this result by specifying the arguments when instantiating 
the RecipeMapper object.

>>> rm = RecipeMapper(adinputs, recipename='/path/to/myrecipes.myreduce')
>>> recipefn = rm.get_applicable_recipe()
>>> recipefn.__name__
'myreduce'

Note that for user supplied recipe libraries and functions, the *mode* is
irrelevant, as it is used for searching the *geminidr* package or other
packages similarly designed.

User-defined recipes
^^^^^^^^^^^^^^^^^^^^

In the case of external (i.e. user-defined) recipes, developers should understand
that in passing a user-defined recipe library to the RecipeMapper, the nominal
mapping algorithm for recipe searches is bypassed and the RecipeMapper will use
the recipe library (module) and path to import the module directly. In these
cases, none of ``mode``, ``tags``, or ``recipe_tags`` is relevant, as the
user-passed recipe library and recipe name are already known. Essentially,
passing a user-defined recipe to the RecipeMapper tells the mapper, "do not
search but use this." In these cases, it is incumbent upon the users and
develoers to ensure that the external recipes specified are actually applicable
to the datasets being processed.

We will now discuss what to do now that we have both a primtives instance and a 
recipe.

Primitives and Recipes, Together at Last
----------------------------------------

As discussed earlier in :ref:`Chapter 3, The Mappers <mapps>`, after
instantiating RecipeMapper and PrimitiveMapper objects with necessary 
parameters, the ``get_applicable_recipe()`` and ``get_applicable_primitives()`` 
methods are respectively called and the returned objects are then combined into 
a nominal function call::

>>> rm = RecipeMapper(adinputs, ...)
>>> pm = PrimitiveMapper(adinputs, ...)
>>> recipe = rm.get_applicable_recipe()
>>> p = pm.get_applicable_primitives()
>>> recipe(p)

That's it. Once the function, ``recipe``, is called with the primitive instance, 
``p``, the pipeline begins execution.

In the context of running ``reduce`` from the command line, the ``Reduce`` class
is responsible for retrieving recipes and primitive sets appropriate to the data
and passing the primitive object as the argument to the recipe function. And while
the ``Reduce`` class provides exception handling during pipeline execution, there
are no such protections at the level of the mapper interfaces. Any exceptions
raised will have to be dealt with by those using the Recipe System at this lower
level interface.

Step-wise Recipe Execution
--------------------------
Since we now understand that a recipe is simply a sequential set of calls on
primitive class methods (the primitives themselves), astute readers will
understand that it is entirely possible to call the recipes steps (primitives)
individually and interactively, and while doing so, inspect the condition of the
data and metdata during step-wise processing.

Starting with an example using a GMOS image, step-wise execution simply becomes
calling the primitives in the same order as the recipe. The example will also
configure a DRAGONS logger object.

The example lays out all import calls and logger configuration, and then shows
an interactive primitive call and inspection of the processed data.

>>> import astrodata
>>> import gemini_instruments
>>> ff = 'S20161025S0111.fits'
>>> ad = astrodata.open(ff)
>>> ad.tags
>>> set(['RAW', 'GMOS', 'GEMINI', 'SIDEREAL', 'UNPREPARED', 'IMAGE', 'SOUTH'])
>>> from gempy.utils import logutils
>>> logutils.config(file_name='rsdemo.log')
>>> from recipe_system.mappers.primitiveMapper import PrimitiveMapper
>>> pm = PrimitiveMapper([ad])
>>> p = pm.get_applicable_primitives()

And begin calling the primitives, the first one is always *prepare*

>>> p.prepare()
   PRIMITIVE: prepare
   ------------------
      PRIMITIVE: validateData
      -----------------------
      .
      PRIMITIVE: standardizeStructure
      -------------------------------
      .
      PRIMITIVE: standardizeHeaders
      -----------------------------
         PRIMITIVE: standardizeObservatoryHeaders
         ----------------------------------------
         Updating keywords that are common to all Gemini data
         .
         PRIMITIVE: standardizeInstrumentHeaders
         ---------------------------------------
         Updating keywords that are specific to GMOS
         .
      .
   .
   [<gemini_instruments.gmos.adclass.AstroDataGmos object at 0x11a12d650>]

As readers can see, the call on the primitive *prepare()* shows the logging
sent to stdout. They will also find the log file, ``rsdemo.log`` in the current
working diretory.

Readers will note the return object. This object is returned both to
the caller, and handled internally by a recipe system decorator function. The
internal handling is not pertinent here, but rather, that the returned object
shown above is a *list* containing the actual AstroDataGmos object(s) that the
primitive class was passed upon construction, but with the *data and metdata in
the current state* at completion of a primitive call. Each primitive returns
this object after completion, allowing users to examine the state of that dataset
at each point in the processing, examine parameters currently set, and set
parameters to new values if desired. But first, one must capture that object on
return, so the previous last call becomes

>>> adobject = p.prepare()
   PRIMITIVE: prepare
   ------------------
      PRIMITIVE: validateData
      -----------------------
      .
      PRIMITIVE: standardizeStructure
      -------------------------------
      .
      PRIMITIVE: standardizeHeaders

>>> ad_prepare = adobject[0]
>>> ad_prepare.data
  array([[  0,   0,   0, ...,   0,   0,   0],
       [  0,   0,   0, ...,   0,   0,   0],
       [  0,   0,   0, ...,   0,   0,   0],
       ...,
       [823, 824, 820, ..., 822, 820, 825],
       [821, 822, 825, ..., 821, 824, 824],
       [823, 819, 823, ..., 205, 204, 203]], dtype=uint16)
>>> ad_prepare.phu.cards['PREPARE']
('PREPARE', '2018-08-24T16:02:39', 'UT time stamp for PREPARE')
>>> ad_prepare.phu.cards['SDZSTRUC']
('SDZSTRUC', '2018-08-24T15:44:08', 'UT time stamp for standardizeStructure')

You can also look at the parameter set for that or any other primitive from the
primtive object itself:

>>> p.params['prepare'].toDict()
OrderedDict([('suffix', '_prepared'), ('mdf', None), ('attach_mdf', True)])
>>> p.params['mosaicDetectors'].toDict()
OrderedDict([('suffix', '_mosaic'), ('tile', False), ('sci_only', False), ('interpolator', 'linear')])

Finally, readers may wonder how one may "see" the recipe the RecipeMapper would
return for the specified data, in order to know the primitives to call and in what
order. This involves using the RecipeMapper just as recipe system does and using
the inspect module to show the function's code.

Continuing the example ...

>>> from recipe_system.mappers.recipeMapper import RecipeMapper
>>> rm = RecipeMapper([ad])
>>> rfn = rm.get_applicable_recipe()
>>> rfn.__name__
'reduce'
>>> import inspect
>>> print inspect.getsource(rfn.__code__)
def reduce(p):
    """
    This recipe performs the standardization and corrections needed to
    convert the raw input science images into a stacked image.
    Parameters
    p : PrimitivesCORE object
        A primitive set matching the recipe_tags.
    """
    p.prepare()
    p.addDQ()
    p.addVAR(read_noise=True)
    p.overscanCorrect()
    p.biasCorrect()
    p.ADUToElectrons()
    p.addVAR(poisson_noise=True)
    p.flatCorrect()
    p.makeFringe()
    p.fringeCorrect()
    p.mosaicDetectors()
    p.alignAndStack()
    p.writeOutputs()
    return

Users can see the next primitive calls, and continue processing the dataset
in a step-wise and interactive manner.

>>> p.addDQ()
   PRIMITIVE: addDQ
   ----------------
   Clipping gmos-s_bpm_HAM_11_12amp_v1.fits to match science data.
   .
[<gemini_instruments.gmos.adclass.AstroDataGmos object at 0x11a12d650>]


.. rubric:: Footnotes

.. [#] See appendix on currently available recipes in geminidr.
