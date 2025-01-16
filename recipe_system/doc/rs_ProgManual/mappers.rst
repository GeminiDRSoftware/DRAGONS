.. interfaces.rst

.. include:: references.txt

.. _mapps:

The Mappers
***********

The mappers are the core of the Recipe System.  The mappers match recipes
and primitives to the data being reduced.

Normally called from the |Reduce| class (which might have been called by the
command line ``reduce``), the mappers will search a data reduction package
(DR package) specified by the argument ``drpkg`` for recipes and primitives
matching the AstroData tags of the data to be processed. (The default ``drpkg``
is DRAGONS ``geminidr`` package.)

The search for matching primitives and recipes is optimized for a certain
directory structure.  Other structures would work though. In the
:ref:`appendix <drpkg>`, we introduce a tool to easily create the data reduction
package structure that will be most efficient.  The reader can also look at
``geminidr`` for a working example.

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


.. _mapperclass:

Mapper Class
============

|Mapper| serves as the base class for the other Recipe System mapper classes.
The base |Mapper| class defines *only* the initialisation function.  Subclasses
inherit and do not override ``Mapper.__init__()``.

It is important to note that only the first AstroData dataset in the list of
input datasets is used to set the tags and the import path of the instrument
package.  It is assumed that all the datasets in the list are of the same type
and will be reduced with the same recipes and primitives.

See the API for |Mapper|.

Regarding the ``recipename`` argument, the default is "_default".
A recipe library should have a module attribute ``_default`` that
is assigned the name of the default function (recipe) to run if that
library is selected. This guarantees that if a library is selected there
will always be a recipe matching ``recipename`` to run.

``recipename`` can also be the name of a specific recipe that will be
expected to be found in the selected recipe library.

To completely bypass the recipe mapping and use a user-provided
recipe library and a recipe within, ``recipename`` is set to

`<(path)library_file_name>.<name_of_recipe_function>`.

Finally, ``recipename`` can be set to be the name of a single primitive.
In that case, the primitive of that name in the mapped primitive set
will be run.


PrimitiveMapper
===============

|PrimitiveMapper| is subclassed on |Mapper| and does *not* override
``__init__()``.  |PrimitiveMapper| implements the primitive search algorithm
and provides one public method on the class: ``get_applicable_primitives()``.

The primitive set search is conducted by comparing the AstroData ``tags``
attribute of the first input dataset to the ``tagset`` attribute of each
primitive class.  The best matched primitive set should, in principle, be found
in an instrument package that matches the instrument descriptor of the input
datasets.  In fact, the way the search is optimized right now, it enforces that
principle.

As the search of instrument primitive classes progresses, modules are
inspected, looking for class objects with a *tagset* attribute. A tagset match
is assessed against all previous matches and the best matching class is retrieved
and instantiated with all the appropriate arguments.

A match requires that the primitive class *tagset* be a subset of the
AstroData *tags* descriptor.  The *best match* is the one with the
largest *tagset*.  If two or more primitive classes return best matches with
the same number of tags in their *tagset*, then the current algorithm will
only return the first *best match* primitive class it has encountered.  It is
therefore very important to be specific with the *tagset* attributes to avoid
such multiple match situation and to ensure only one true best match.

The ``get_applicable_primitives()`` method returns this instance of the best
match primitive class. The object returned will be the actual instance and
usable as such as an argument to a recipe function.  The list of AstroData
objects given as input to PrimitiveMapper is used to instantiate the
chosen primitive class.

The instantiation of the primitive class by ``get_applicable_primitives()``
implies an API requirement on the Primitive class.  It must be possible to
instantiate a primitive class with the following call::

    PrimitiveClassName(self.adinputs, mode=self.mode, ucals=self.usercals,
        uparms=self.userparams, upload=self.upload, config_file=self.config_file)

where ``self`` is an instantiated PrimitiveMapper.


RecipeMapper
============

|RecipeMapper| is subclassed on |Mapper| and does *not* override ``__init__()``.
|RecipeMapper| implements the recipe search algorithm and provides one
public method on the class: ``get_applicable_recipe()``.

The recipe library search is conducted by comparing the AstroData ``tags``
attribute of the first input dataset to the ``recipe_tags`` module attribute
of each recipe library in the ``mode`` subdirectory of the instrument package.
The best matched recipe library should in principle be found in
an instrument package that matches the instrument descriptor of the input
datasets.  In fact, the way the search is optimized right now, it enforces that
principle.

The ``mode`` narrows the recipe search in the instrument package to the
corresponding subdirectory, while the AstroData tags are used to locate the
desired recipe library within that ``mode`` subdirectory. As the search of
instrument recipe modules (libraries) progresses, modules are inspected,
looking for a ``recipe_tags`` attribute. A recipe tags match is assessed against
all previous matches and the best matching recipe library is imported.
The *default* or the named recipe function is retrieved from the recipe
library.

A match requires that the *recipe_tags* be a subset of the
AstroData *tags* descriptor.  The *best match* is the one with the
largest matching subset of the data's tags attribute.  If two or more
recipe libraries return best matches with the same number of tags in
their *recipe_tags*,
then the current algorithm will only return the first *best match* recipe
library it has encountered.  It is therefore very important to be specific
with the *tagset* attributes to avoid such multiple match situation and to
ensure only one true best match.

The ``get_applicable_recipe()`` method returns this *best match* recipe
function to the caller.  This is a function object and is callable.

The Handling of ``recipename``
==============================
The input argument ``recipename`` is multiplexed. The default value is the
string *default* that matches a module attribute in the recipe library
identifying which of the recipes is to be considered the default.

``recipename`` may be set to the name of a specific recipe (function)
within a recipe library. This will override the default setting. Of course,
the function must be present in the *best match* recipe library.

The ``recipename`` can also specify a user-provided recipe.  The syntax for
this form is ``<recipe_library>.<recipe>``.  If the <recipe_library> string
does not contain the full path to the module, the current directory is assumed
to be the location of the library.  The |RecipeMapper| first tries to find
such a recipe. If it is not found, then the mapper begins the process of
searching for the *best match* recipe in the data reduction package.

Finally, the ``recipename`` can be the name of a primitive rather than the
name of a recipe (eg. ``reduce N20120212S0012.fits -r display``).  In that
case, the |RecipeMapper| will fail to find a matching recipe. Recognition
of the string as a valid primitive is done in the ``runr()`` method of  |Reduce|.
The primitive must be one of the primitives from the *best match* primitive set.
