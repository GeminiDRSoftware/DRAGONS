.. drpkg.rst

.. include:: ../references.txt

.. _drpkg:

Data Reduction Packages
***********************
In the context of the Recipe System, a *Data Reduction Package* is a python
package containing one or more subpackages, each providing instrument-specific
sets of data processing primitive classes and recipe libraries. These packages provide attributes on certain components of the
package, which make them discoverable by the Recipe System. Developers are
entirely free to build their own data reduction package, or "dr-package."

As stated at :ref:`the beginning of Chapter 4 <iface>`, the default and only data
reduction package provided by DRAGONS is ``geminidr``. This package is included
in the DRAGONS distribution. Unless specifed otherwise, it is the
``geminidr`` package that serves targets for the Recipe System mappers. Readers are
encouraged to examine the ``geminidr`` package to familiarize themselves with
components.



Building a new Data Reduction Package
=====================================
The Recipe System mappers require the primitives and the recipes to be
organized under a specific directory structure.  To help build that structure
the Recipe System provides a script called ``makedrpkg`` that we will
introduce later in this section.  First, we address the requirements.

* The Data Reduction (DR) package **must** be importable.  (It must have a
  ``__init__.py``.)
* The DR package **must** be found in one of the ``sys.path`` directories.
* The instrument packages must be found at the first subdirectory level in the
  DR package.
* The instrument packages (directory name) must be named after the instrument
  they associate with.  The instrument package name **must** match the lower
  case version of the AstroData descriptor ``instrument`` for the data it
  supports.
* The recipes must be in a subdirectory of the instrument package.  That
  directory **must** be named ``recipes``.  That name is hardcoded in the
  moduleattribute ``RECIPEMARKER`` in ``utils.mapper_utils``.
* The recipes **must** be assigned a ``mode`` (one of "sq", "qa", "ql").
* The mode-specific recipes **must** be located in a subdirectory of
  ``recipes``.  That directory **must** be named to match the mode.
* The ``recipes`` directory and the *mode* directories but all have an
  ``__init__.py`` in them and be importable.


The directory structure can be created by hand but to simplify the process
and avoid mistakes, it is recommended to use the ``makedrpkg`` script
provided with the Recipe System.  The script is used from a normal terminal,
not from Python. Here is a few usage examples.

Get help::

    % makedrpkg -h

Create ``mydrpkg`` DR package with a ``coolinstrument`` instrument package
and a ``sq`` mode subdirectory::

    % makedrpkg mydrpkg -i coolinstrument -m sq

Same as above but with both ``sq`` and ``qa`` modes.  Note that if a directory
already exists it will just be skipped.

::

    % makedrpkg mydrpkg -i coolinstrument -m sq qa

Add two instrument packages with ``qa`` mode::

    % makedrpkg mydrpkg -i instA instB -m qa

Add a ``sq`` mode to an existing instrument package::

    % makedrpkg mydrpkg -i instA -m sq


Once you have that structure in place, the primitives and the parameters
modules go in the instrument package main directory, and the recipes in the
``recipes/<mode>`` directory.


Using a third-party Data Reduction Package
==========================================
To activate a specific DR package, the ``drpkg`` attribute or option (depending
on what is being used) needs to be set.   The default setting is ``geminidr``.

From the ``reduce`` command line tool
-------------------------------------
From the ``reduce`` command line tool one uses the ``--drpkg`` option.  For
example::

    % reduce *.fits --drpkg mydrpkg

From the |Reduce| class API
---------------------------
When using the |Reduce| class API, the attribute to set is ``drpkg``.  For
example::

    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> reduce = Reduce()
    >>> reduce.drpkg = 'mydrpkg'


From the mappers
----------------
When using the mappers directly, again the attribute to set in either mapper,
|PrimitiveMapper| or |RecipeMapper|, is ``drpkg``.  For example::

    >>> from recipe_system.mappers.primitiveMapper import PrimitiveMapper
    >>> from recipe_system.mappers.recipeMapper import RecipeMapper
    >>> tags = ad.tags
    >>> instpkg = ad.instrument(generic=True).lower()
    >>> pmapper = PrimitiveMapper(tags, instpkg, drpkg='mydrpkg')
    >>> rmapper = RecipeMapper(tags, instpkg, drpkg='mydrpkg')




Requirements for Primitives and Recipes
=======================================
Instructions on how to write primitives and recipes is beyond the scope of this
manual.  However, the Recipe System does impose some requirements on the
primitives and the recipes.  We review them here.

Requirements on Recipes
-----------------------
* A recipe library must contain a module attribute named ``recipe_tags`` that
  contains a Python set of the AstroData tags applicable to the library.
* A recipe library must contain a module attribute named ``default`` that
  sets the name of the default recipe for this library.  The ``default``
  attribute needs to be set below the recipe function definition for Python
  to pick it up.
* A recipe signature must accept the primitive set as the first argument with
  no other "required" arguments.  Other arguments must be optional.

Requirements on Primitives
--------------------------
The requirements on the primitives are highly Gemini centric for the moment.

* A primitive class must inherit ``PrimitiveBASE`` class defined in
  ``geminidr/__init__py``, or bring a copy of it in a third-party DR package.
* A primitive class must be decorated with the ``parameter_override``
  decorator located in ``recipe_system.utils.decorators``.
* The Gemini fork of LSST pexconfig package, ``gempy.library.config``, must be
  used to handle input parameters.
* The signature of the ``__init__`` of primitive class should be::

    def __init__(self, adinputs, **kwargs):

* The signature of a primitive, a method of a primitive classe should be::

    def primitive_name(self, adinputs=None, **params)

* Primitive parameter must be defined in a class matching the name of the
  primitive followed by "Config".  For example::

    class primitive_nameConfig(any_other_parameter_class_to_inherit):
        param1 = config.Field("Description of param", <type>, <default_value>)

* Each primitive class must define a ``tagset`` attribute containing a ``set``
  of AstroData tags identifying which data this primitive class is best suited
  for.  The tags are string literals.  This ``tagset`` attribute is what
  the primitive mapper uses to find the most appropriate primitive class.

Here is an example putting most of the above requirements to use::

    from geminidr.core import Image, Photometry
    from .primitives_gmos import GMOS
    from . import parameters_gmos_image
    from recipe_system.utils.decorators import parameter_override

    @parameter_override
    class GMOSImage(GMOS, Image, Photometry):

        tagset = set(["GEMINI", "GMOS", "IMAGE"])

        def __init__(self, adinputs, **kwargs):
            super().__init__(adinputs, **kwargs)
            self._param_update(parameters_gmos_image)

        def some_primitive(self, adinputs=None, **params):
            [...]
            return adinputs

It is technically possible to decide not to use the ``parameter_override``
decorator.  In that case, there will be no transparent passing of the input
AstroData objects, all primitives will have to have their parameter explicitely
defined when called, the primitive class will have to have exact signature
the ``get_applicable_primitives`` uses when initializing the class.  This is
not a mode we have experimented with and there might be additional limitations.

The primitive class signature must be able to accept this instantiation call::

   primitive_actual(self.adinputs, mode=self.mode, ucals=self.usercals,
                    uparms=self.userparams, upload=self.upload)

    adinputs: Python list of AstroData objects
    mode:     One of 'sq', 'qa', or 'ql'
    ucals:    Python dict with format
                 {(<data_label>, <type_of_calib>): <calib_filename>},
                 one key-value pair for each input, with the type of
                 calibration matching one from the keys in
                 cal_service.caldb.REQUIRED_TAG_DICT.
    uparms:   Python dict with format ``{'<prim>:<param>': <value>}``
    upload:   Python list of any combination of 'calibs', 'metrics', or
                 'science'.


Requirement to use AstroData
----------------------------
The Recipe System expects to work on AstroData objects.  In particular, it
requires the ``tags`` to be defined and the ``instrument()`` descriptor to
be defined.

Therefore, for data from a new, still unsupported instrument, the first step
is to write the AstroData configuration layer.  In DRAGONS, the AstroData
configuration layers are found in the package ``gemini_instruments``.  The
convention is to name the modules with the tags and descriptor ``adclass.py``.
This is just a convention.

For more information on the AstroData configuration, see the |astrodataprog|.
