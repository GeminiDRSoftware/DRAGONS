.. interfaces.rst
.. include overview

.. _mapps:

The Mappers
***********

**map** (verb) - *(OED)*

1. *Associate (a group of elements or qualities) with an equivalent group, 
   according to a particular formula or model.*
2. *Associate each element of (a set) with an element of another set.*

----

In a nominal pipeline context, the mappers receive input data and parameters from 
the ``Reduce`` class, either through the ``reduce`` command or the class's API. This
document describes how to import and use the mapper classes programmatically. The 
mapper classes are the main component of the Recipe System and serve as arbiters 
between the input data and parameters and the instrument packages defined
by the ``drpkg`` parameter. This parameter defaults to ``geminidr``, which is the
Gemini Observatory's data reduction ('dr') package under *gemini_python*.

The mapper classes implement search/match/capture algorithms optimized for the 
instrument packages defined in ``geminidr`` under a *gemini_python* installation. 
The algorithm exploits attributes of an ``AstroData`` instance, determines the 
applicable instrument package (e.g. 'niri', 'gnirs', 'gmos', etc.) defined under 
``geminidr``, then conducts a search of that package, looking for specific 
attributes: a class attribute called `tagset` defined on discoverable 
primitive classes, and a module attribute defined in recipe library files, 
called `recipe_tags`. The mappers look for the "best" match between an AstroData 
object's tags and the tags defined on recipe libraries and primitive classes, 
where best here requires that the target tagsets are a *complete subset* of the 
larger set of tags defined for the input dataset.

Mapper
======

Mapper (in ``recipe_system.mappers.baseMapper``) serves as the base class for all
other `Recipe System` mapper classes. The base Mapper class defines *only* the 
initialisation function, which sets important instance attributes, such as the 
applicable instrument package name (based on instrument name), and a recipe name. 
This base class *__init__* function receives all data and parameters passed by 
either ``Reduce`` or other caller. As readers may infer from the 
:ref:`Mapper class diagram <mappercls>` below, the Mapper initializer determines 
certain instance attributes `from` the passed input datasets provided by the list, 
``adinputs``.

For instance, the mapper attribute, ``pkg``, which is used as the description of 
the applicable instrument package in *geminidr*, is derived from an ``AstroData`` 
instance "descriptor," ``instrument()``, which returns the actual instrument used 
for the observation. (While details about AstroData classes are beyond the scope 
of this document, readers are encouraged to consult the AstroData documents listed 
in Chapter 2, :ref:`Related Documents <refdocs>`.)

The Mapper classes receive a required list of input datasets and some, or all, of 
the following arguments. Of the arguments listed below, only ``adinputs``, 
``recipename``, and ``context`` are used by currently defined mappers to find 
recipes and primitive classes. All other arguments listed are passed to, and used 
by, the selected primitive class.

.. _mappercls:

.. figure:: images/mpscls.png
   :scale: 80

   Mapper subclasses inherit and do not override Mapper.__init__().

PrimitiveMapper
===============

PrimitiveMapper (in ``recipe_system.mappers.primitiveMapper``) is subclassed on
Mapper. PrimitiveMapper implements the primitive search algorithm and provides one 
(1) public method on the class: ``get_applicable_primitives()``.

 **Class PrimitiveMapper** `(adinputs, context=['sq'], drpkg='geminidr', recipename='default', usercals=None, uparms=None, upload_metrics=False)`

   adinputs
     A `list` of input AstroData objects (required).
   context
     A `list` of all passed contexts. This defines which recipe set to use but
     may also contain other string literals, which are passed on to primitives
     that may interpret certain string elements for their own purposes.
     Default is ['sq']. The context parameter is discussed in greater detail in
     the next chapter in :ref:`Selecting Recipes with RecipeMapper <rselect>`.
   drpkg
     A `string` indicating the name of the data reduction package to map. Default
     is 'geminidr'. The package *must* be importable and should provide instrument
     packages of the same form as defined under *geminidr*.
   recipename
     A `string` indicating the recipe to use for processing. ``recipename`` may
     be a system or external recipe name, as passed by a ``reduce`` command with 
     ``-r`` or ``--recipe``, or set directly by a caller. This string may also
     be an *explicitly named primitive function*. Otherwise, recipename is 
     'default', which may be an actual function or a reference to a named recipe 
     function defined in a recipes library. In *gemindr* recipe packages,
     defaults are references to other defined recipe functions.
   usercals
     A `dictionary` of user provided calibration files, keyed on cal type.
     E.g., {'processed_bias': 'foo_bias.fits'}
   uparms
     A `list` of tuples representing user parameters passed via command line or 
     other caller. Each may have a specified primitive.
     E.g., [('foo','bar'), ('tileArrays:par1','val1')]
   upload_metrics
     A `boolean` indicating whether to send QA metrics to fitsstore.
     Default is False. Only the *geminidr* primitives, measureBG, measureCC, 
     and measureIQ produce these metrics.

 Public Methods

  **get_applicable_primitives** (self)

     `Parameters`

       None

     `Return`

      `<instance>` of a primitive class.


The "applicable" primitives search is conducted by employing only one parameter 
passed to the class initializer, the astrodata *tagset* attribute of the input 
dataset(s). The *tagset* is used to find the appropriate primitive class. For
real data, i.e., data taken with an actual instrument, the applicable primitives 
class will always be found in an instrument package, as opposed to the generic 
primitive classes of the *geminidr* primitive class hierarchy.

As the search of instrument primitive classes progresses, modules are 
introspected, looking for class objects with a *taget* attribute. A tagset match 
is assessed against all previous matches and the best matching class is retrieved 
and instantiated with all the appropriate arguments received from ``Reduce``, or
set as instance attributes through the class API.

The ``get_applicable_primitives()`` method returns this instance of the best 
match primitive class to the caller. The object returned will be the actual 
instance and usable as such.

It will be this primitive instance that can then be passed to the "applicable"
recipe as returned by the RecipeMapper.

RecipeMapper
============

RecipeMapper (in ``recipe_system.mappers.recipeMapper``) is subclassed on
Mapper and does *not* override ``__init__()``. RecipeMapper implements the 
recipe search algorithm and provides one (1) public method on the class:
``get_applicable_recipe()``.

 **Class RecipeMapper** `(adinputs, context=['sq'], drpkg='geminidr', recipename='default', usercals=None, uparms=None, upload_metrics=False)`

   adinputs
     A `list` of input AstroData objects (required).
   context
     A `list` of all passed contexts. This defines which recipe set to use but
     may also contain other string literals, which are passed on to primitives
     that may interpret certain string elements for their own purposes. 
     Default is ['sq']. The context parameter is discussed in greater detail in
     the next chapter in :ref:`Selecting Recipes with RecipeMapper <rselect>`.
   drpkg
     A `string` indicating the name of the data reduction package to map. Default
     is 'geminidr'. The package *must* be importable and should provide instrument
     packages of the same form as defined under *geminidr*.
   recipename
     A `string` indicating the recipe to use for processing. ``recipename`` may
     be a system or external recipe name, as passed by a ``reduce`` command with 
     ``-r`` or ``--recipe``, or set directly by a caller. This string may also
     be an *explicitly named primitive function*. Otherwise, recipename is 
     'default', which may be an actual function or a reference to a named recipe 
     function defined in a recipes library. In *gemindr* recipe packages,
     defaults are references to other defined recipe functions.
   usercals
     A `dictionary` of user provided calibration files, keyed on cal type.
     E.g., {'processed_bias': 'foo_bias.fits'}
   uparms
     A `list` of tuples representing user parameters passed via command line or 
     other caller. Each may have a specified primitive.
     E.g., [('foo','bar'), ('tileArrays:par1','val1')]
   upload_metrics
     A `boolean` indicating whether to send QA metrics to fitsstore.
     Default is False. Only the *geminidr* primitives, measureBG, measureCC, 
     and measureIQ produce these metrics.

 Public Methods

  **get_applicable_recipe** (self)

     Parameters

       None

     Return

      `<type 'function'>` A function defined in an instrument package recipe library.


The "applicable" recipe search is conducted by employing two parameters passed 
to the class initializer, the *context* and the astrodata *tagset* attribute of 
the input dataset(s). The *context* narrows the recipe search in the instrument 
package, while the *tagset* is used to locate the desired recipe library. This 
library is imported and the named recipe function retrieved. The 
``get_applicable_recipe()`` method returns this recipe function to the caller. 
This will be the actual function object and will be callable. 

As the search of instrument recipe modules (libraries) progresses, modules are 
introspected, looking for a *recipe_tags* attribute. A recipe tags match is 
assessed against all previous matches and the best matching recipe library is 
imported with all the appropriate arguments received from ``Reduce``, or set as 
instance attributes through the class API.

Because the RecipeMapper class must be responsive to a number of possible 
forms a recipe name may be take as specified by clients, such as the ``reduce``
command line tool and the ``Reduce`` class, the RecipeMapper first examines the 
recipe name to see if it can be found as a member of an external recipe library, 
i.e., not defined under the *geminidr* package. If not, this mapper class then 
begins the process of searching for the correct ("applicable") recipe in 
*geminidr* under the appropriate instrument package.



