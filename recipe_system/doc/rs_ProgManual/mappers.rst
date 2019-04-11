.. interfaces.rst

.. include:: references.txt

.. _mapps:

The Mappers
***********

The mappers are the heart of the Recipe System.  The mappers associate recipes
and primitives to the data being reduced.

Normally called from the |Reduce| (which might have been called by the command
line ``reduce``), the mappers will search a data reduction package (DR package)
specified by
the argument ``drpkg`` for recipes and primitives matching the AstroData tags
of the data to be processed.  (The default ``drpkg`` is DRAGONS ``geminidr``
package.)

The search for matching primitives and recipes is optimized for a certain
directory structure.  Other structure would work though.  In the appendix,
we introduce a tool to easily create the data reduction package structure that
will be most efficient.  The reader can also look at ``geminidr`` for a
working example.
.. todo:: add reference to appendix.

The mapping algorithm uses attributes of an ``AstroData`` instance to first
determines the applicable instrument package defined under the DR package,
for example ``gmos`` in ``geminidr``. Then the algorithm conducts a search of
that instrument package looking for specific attributes: a class attribute
called ``tagset`` defined in discoverable primitive classes, and a module
attribute, called ``recipe_tags``, defined in recipe library modules.
The mappers look for the "best" match between an AstroData
object's tags and the tagset defined in recipe libraries and primitive classes.
The best match requires that the primitive and recipe tagsets are a
*complete subset* of the larger set of tags defined for the input dataset. The
association with the greatest number of matched tags wins the test.


Mapper Class
============

|Mapper| serves as the base class for the other Recipe System mapper classes.
The base |Mapper| class defines *only* the initialisation function.  Subclasses
inherit and do not override ``Mapper.__init__()``.

It is important to note that only the first AstroData dataset in the list of
input datasets is used to set the tags and the import path of the instrument
package.  It is assumed that all the datasets in the list are of the same type
and will be reduced with the same recipes and primitives.

.. todo:: below in drpkg, add reference to appendix


**Class Mapper** *(adinputs, mode='sq', drpkg='geminidr', recipename='default', usercals=None, uparms=None, upload=None)*

   Arguments to __init__
    adinputs
        A ``list`` of input AstroData objects (required).
    mode
        A ``str`` indicating mode name. This defines which recipe set to use.
        Default is ``'sq'``. See the
        :ref:`mode definition <modedef>` for addtional information.
    drpkg
        A ``str`` indicating the name of the data reduction package to inspect.
        The default is ``geminidr``. The package *must* be importable and
        should provide instrument packages of the same form as defined in
        appendix.
    recipename
        A ``str`` indicating the recipe to use for processing. This will
        override the mapping in part or in whole.  The default is "default".
        A recipe library should have a module attribute ``default`` that
        is assigned the name of the default function (recipe) to run if that
        library is selected. This guarantee that if a library is selected there
        will always be a recipe matching ``recipename`` to run.

        ``recipename`` can also be the name of a specific recipe that will be
        expected to be found in the selected recipe library.

        To completely bypass the recipe mapping and use a user-provided
        recipe library and a recipe within, ``recipename`` is set to
        `<(path)library_file_name>.<name_of_recipe_function>`.

        Finally, ``recipename`` can be set to be the name of a single primitive.
        In that case, the primitive of that name in the mapped primitive set
        will be run.
    usercals
        A `dictionary` of user-specified calibration files, keyed on
        calibration type.
        E.g., ``{'processed_bias':'foo_bias.fits'}``
    uparms
        A `list` of tuples representing user parameters passed via command line or
        other caller. Each may have a specified primitive.
        E.g., [('foo','bar'), ('tileArrays:par1','val1')]
    upload
        A `list` of strings indicating the processing products to be uploaded to
        fitsstore. For example, in running the QA pipeline, upload = ['metrics'],
        where the *geminidr* primitives, measureBG, measureCC, and measureIQ produce
        these QA metrics. Default is None.

   Attributes
    pkg
        blah
    dotpackage
        blah
    userparams
        blah


PrimitiveMapper
===============

|PrimitiveMapper| is subclassed on |Mapper| and does *not* override ``__init__()``.
|PrimitiveMapper| implements the primitive search algorithm and provides one (1)
public method on the class: ``get_applicable_primitives()``.

 **Class PrimitiveMapper** `(adinputs, mode='sq', drpkg='geminidr', recipename='default', usercals=None, uparms=None, upload=None)`

   adinputs
     A `list` of input AstroData objects (required).
   mode
     A `string` indicating mode name. This defines which recipe set to use.
     Default is 'sq'. The mode parameter is discussed in greater detail in
     the next chapter in :ref:`Selecting Recipes with RecipeMapper <rselect>`.
   drpkg
     A `string` indicating the name of the data reduction package to map. Default
     is 'geminidr'. The package *must* be importable and should provide instrument
     packages of the same form as defined under *geminidr*.
   recipename
     A `string` indicating the recipe to use for processing. ``recipename`` may
     be a system or external recipe name, as passed by a ``reduce`` command with 
     ``-r`` or ``--recipe``, or set directly by a caller. This string may also
     be an *explicitly named primitive function*. Otherwise, ``recipename`` is
     'default', which may be an actual function or a reference to a named recipe 
     function defined in a recipes library. In *gemindr* recipe packages,
     defaults are references to other defined recipe functions.
   usercals
     A `dictionary` of user provided calibration files, keyed on cal type.
     E.g., ``{'processed_bias':'foo_bias.fits'}``
   uparms
     A `list` of tuples representing user parameters passed via command line or 
     other caller. Each may have a specified primitive.
     E.g., [('foo','bar'), ('tileArrays:par1','val1')]
   upload
     A `list` of strings indicating the processing products to be uploaded to
     fitsstore. For example, in running the QA pipeline, upload = ['metrics'],
     where the *geminidr* primitives, measureBG, measureCC, and measureIQ produce
     these QA metrics. Default is None.

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
class will always be found in an instrument package, as opposed to the more generic 
primitive classes of the *geminidr* primitive class hierarchy.

As the search of instrument primitive classes progresses, modules are 
introspected, looking for class objects with a *tagset* attribute. A tagset match 
is assessed against all previous matches and the best matching class is retrieved 
and instantiated with all the appropriate arguments received from |Reduce|, or
set as instance attributes through the class API.

The ``get_applicable_primitives()`` method returns this instance of the best 
match primitive class. The object returned will be the actual instance and usable
as such. It will be this primitive instance that can then be passed to the
"applicable" recipe as returned by the RecipeMapper.

RecipeMapper
============

|RecipeMapper| is subclassed on |Mapper| and does *not* override ``__init__()``.
|RecipeMapper| implements the recipe search algorithm and provides one (1)
public method on the class: ``get_applicable_recipe()``.

 **Class RecipeMapper** `(adinputs, mode='sq', drpkg='geminidr', recipename='default', usercals=None, uparms=None, upload=None)`

   adinputs
     A `list` of input AstroData objects (required).
   mode
     A `string` indicating mode name. This defines which recipe set to use.
     Default is 'sq'. The mode parameter is discussed in greater detail in
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
     E.g., ``{'processed_bias':'foo_bias.fits'}``
   uparms
     A `list` of tuples representing user parameters passed via command line or 
     other caller. Each may have a specified primitive.
     E.g., [('foo','bar'), ('tileArrays:par1','val1')]
   upload
     A `list` of strings indicating the processing products to be uploaded to
     fitsstore. For example, in running the QA pipeline, upload = ['metrics'],
     where the *geminidr* primitives, measureBG, measureCC, and measureIQ produce
     these QA metrics. Default is None.

 Public Methods

  **get_applicable_recipe** (self)

     `Parameters`

       None

     `Return`

      `<type 'function'>` A function defined in an instrument package recipe library.


The "applicable" recipe search is conducted by employing two parameters passed 
to the class initializer, the *mode* and the astrodata *tagset* attribute of 
the input dataset(s). The *mode* narrows the recipe search in the instrument 
package, while the *tagset* is used to locate the desired recipe library. This 
library is imported and the named recipe function retrieved. The 
``get_applicable_recipe()`` method returns this recipe function to the caller. 
This will be the actual function object and will be callable. 

As the search of instrument recipe modules (libraries) progresses, modules are 
introspected, looking for a *recipe_tags* attribute. A recipe tags match is 
assessed against all previous matches and the best matching recipe library is 
imported with all the appropriate arguments received from |Reduce|, or set as
instance attributes through the class API.

Because the RecipeMapper class must be responsive to a number of possible 
forms a recipe name may be take as specified by clients, such as the ``reduce``
command line tool and the |Reduce| class, the |RecipeMapper| first examines the
recipe name to see if it can be found as a member of an external recipe library, 
i.e., not defined under the |geminidr| package. If not, this mapper class then
begins the process of searching for the correct ("applicable") recipe in 
|geminidr| under the appropriate instrument package.
