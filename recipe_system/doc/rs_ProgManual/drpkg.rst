.. drpkg.rst
.. include interfaces
.. include overview

Data Reduction Packages
***********************
In the context of the Recipe System, a "data reduction package" is a python package
containing one or more subpackages -- each providing instrument-specific sets of 
data processing primitive classes and recipe libraries for that instrument.
These packages provide  attributes on certain components of the package, which make
them discoverable by the Recipe System. Developers are entirely free to build their
own data reduction package, or "dr-package."

As stated at :ref:`the beginning of Chapter 4<iface>`, the default and only data
reduction package provided by *gemini_python* is *geminidr*. This package is included
in the *gemini_python* distribution. Unless specifed otherwise, it is the
*geminidr* package that serves targets for the Recipe System mappers. Readers are
encouraged to examine the *geminidr* package to familiarize themselves with
components, though, in truth, the top-level package structure is quite simple.

.. _drpkg:

Building a new "dr-package"
===========================
Developers can build and write one or more of their own "dr-packages." There may be
any number of reasons why one might wish to do this, but we have found it handy for
testing purposes; one might want to build and test a new data reduction package
independent of *geminidr*. This document won't argue against that -- it is entirely
possible to place any new instrument packages under *geminidr* without undue side
effects -- but rather, demonstrate the "how to" of setting up a "dr-package".
And it's really very simple.

The first requirement is that your new *dr-package* `must` be directly importable by
python in the same way *geminidr* is directly importable::

  >>> import geminidr
  >>> geminidr
  <module 'geminidr' from '../gemini_python/geminidr/__init__.py'>

Which is to say, the new package must be importable from some path defined either in
``sys.path`` or a user's PYTHONPATH environment variable. For convenience, this
document suggests that you can simply place your own *dr-package* under
*gemini_python* but this is not required. What `is` required is that your new
*dr-package* must be importable in this same way. Now, let's make that package.

We start by simply making a directory, naming it as our new data reduction
package. For convenience, we shall do this under *gemini_python*, making it
"parallel" with *geminidr*. We shall show each step, including making the "empty"
__init__.py files for the subpackages. Readers and developers are free to make 
this directory structure in any way they like, but here we show the build
using common shell commands::

    ../gemini_python $ mkdir testbed
    ../gemini_python $ cd testbed
    ../gemini_python $ touch __init__.py
    ../gemini_python $ mkdir new_instrument
    ../gemini_python $ cd new_instrument
    ../gemini_python $ touch __init__.py
    ../gemini_python $ touch new_instrument_primitives.py
    ../gemini_python $ touch new_instrument_parameters.py
    ../gemini_python $ mkdir recipes
    ../gemini_python $ cd recipes
    ../gemini_python $ touch __init__.py
    ../gemini_python $ mkdir new_context
    ../gemini_python $ touch new_context/__init__.py

At this point, you now have the basic template of a *dr-package*, named ``testbed``,
laid out and that contains one (1) instrument package for an instrument named,
"new_instrument." Your new *dr-package* appears as follows::

  testbed:
      __init__.py
      new_instrument:
          __init__.py
          new_instrument_parameters.py
          new_instrument_primitives.py
          recipes:
              __init__.py
              new_context:

A "dr-package" build function
=============================
For those who would rather not mimic a bunch of command line calls on cd, mkdir, and
touch (tedious to be sure!), the following two (2) functions will provide dr-package
building functionality with no need to type things on a command line (or other
interface)::

 from os import makedirs
 from os.path import join

 def touch(fname):
     with open(fname, 'a'):
         return

 def mkpkg(pkgname, instr, ctx=None):
     if ctx is None:
         ctx = 'sq'

     rpath = join(pkgname, instr, 'recipes')
     ctxpath = join(pkgname, instr, 'recipes', ctx)
     makedirs(join(ctxpath))
     touch(join(pkgname, '__init__.py'))
     touch(join(pkgname, instr, '__init__.py'))
     touch(join(rpath, '__init__.py'))
     touch(join(ctxpath, '__init__.py'))
     return ctxpath

Place this in your own importable python module and call the ``mkpkg()`` function
with the component names you like. The function will return the full path to last
componenet made, in this case, the full path to the context, ``ctxpath``. If you
pass no context name, a default of ``'sq'`` is provided. Remember, the ``recipes``
package *must* provide one or more further categorisations called "contexts."

>>> import mkpkg
>>> mkpkg.mkpkg('newfoo', 'instr_XXX', 'myctx')
'newfoo/instr_XXX/recipes/myctx'
>>> mkpkg.mkpkg('newfoo', 'instr_XXX', 'sq')
'newfoo/instr_XXX/recipes/sq'

As you can see in the function, all __init__.py package files are also made during
package build. [#f1]_  You can repeatedly call this function and pass it different
values for the context (``ctx=``), instrument name (``instr``), etc., and the
function will keep building up your new dr-package with each new call.

.. rubric:: Footnotes

.. [#f1] We are building packages with the nominal package file, __init__.py,
         which is fine for both Py2.x and Py3.x. These are called "regular
	 packages." Python 3.3 introduces the concept of *Implicit Namespace 
	 Packages*, which allows packages to be recognized as such without the
	 presence of __init__.py files. The documentation assures us that
	 *"there is no functional difference between [a namespace package] and a
	 regular package."* Using __init__.py files in packages provides
	 compatibility across Py2.x and Py3.x.


Accessing a new "dr-package"
============================
At this point, we now have an instrument package defined in a new *dr-package* named,
``testbed`` and this can be requested by the many ways one can set this attribute.
For example, with the ``reduce`` command::

  $ reduce --drpkg testbed --context new_context ...

On the Reduce class [#f2]_:

>>> from recipe_system.reduction.coreReduce import Reduce
>>> reduce = Reduce()
>>> reduce.drpkg
'geminidr'
>>> reduce.drpkg = 'testbed'
>>> reduce.context = 'new_context'

From the mappers interfaces:

>>> from recipe_system.mappers.primitiveMapper import PrimitiveMapper
>>> from recipe_system.mappers.recipeMapper import RecipeMapper
>>> pm = PrimitiveMapper([ad], drpkg='testbed')
>>> rm = RecipeMapper([ad], drpkg='testbed', context=['new_context'])

.. rubric:: Footnotes

.. [#f2] Readers unfamiliar with ``reduce`` and/or the Reduce class interfaces,
         should consult the `Reduce and Recipe System User Manual,`
         Doc. ID: PIPE-USER-109_RSUsersManual, 2017, as cited in :ref:`Sec. 2.1,
         Reference Documents <refdocs>`.

As readers examine *geminidr* instrument packages, they will notice some or many
of these have a ``lookups/`` directory. This is convention and the standard place
*geminidr* organizes instrument-specific lookup tables, such as tables for detector
array gaps, geometries, other definition files, etc.. The absence or presence of
``lookups/`` is immaterial to the Recipe System and can be present at the
convenience of the developer.

At this point, it is incumbent upon the developer to provide the primitive classes
and recipes they wish to define. You are free to inherit or use any primitive and
parameter classes from *geminidr* or to not inherit anything at all. You can also
make use of any and all function libraries from the larger *gemini_python*
distribution.

This document does not purport to offer instruction on how to write primitive
classes and methods -- this is beyond the scope of the current document. This
document *will* specify and describe attributes that must appear in a defined
primitive class and recipe libraries if the Recipe System is to handle your newly
defined *dr-package* and underlying instrument package.

1) You must use the ``parameter_override`` decorator for the new primitive class.
   This decorator handles parameters for all methods on the decorated class. This
   decorator is located in ``recipe_system.utils.decorators.``

2) You must define a ``tagset`` as a class attribute on the new primitive class.
   This ``tagset`` must be a python *set* object and it must provide a set of
   string literals that "describe" the kind of data to which the primitive is
   applicable. This tagset allows the Recipe System to assess discoverable
   primitives.

3) In recipe files defined under ``recipes/<context>/``, define a ``recipe_tags``
   attribute at the module/file level. These tags define the kinds of data for
   which the defined recipes are appropriate. Readers are encouraged to examine
   the contents of any *geminidr* instrument package recipes and contexts to get
   a sense of these recipe libraries.

Readers may wish to review the relevent sections of the
:ref:`Chapter 2, Overview <overview>` and specifically,
:ref:`Sec. 2.5.3, Instrument Packages <ipkg>`. 

As an example primitive, here is what we see on the GMOS class::

  from recipe_system.utils.decorators import parameter_override

  @parameter_override
  class GMOS(Gemini, CCD):
      tagset = set(["GEMINI", "GMOS"])

      def __init__(self, adinputs, **kwargs):  
          [ ... ]

Those are the requirements of the Recipe System for any new *dr-package*.

But there is more. Developers need also to define and configure an *astrodata*
class for this new instrument. This, too, is beyond the scope of this document,
but since the Recipe System uses the grammar of the astrodata abstraction layer,
it is incumbent upon developers that any new instrument, whether in *geminidr* or
elsewhere, is that any new instrument defines a set of *astrodata* tags for the
new instrument and one (1) descriptor must be defined. This descriptor shall be
called ``instrument()`` and it shall return the actual name of the instrument.
In our example case, this instrument descriptor shall return the string literal
``new_instrument``::

  >>> ad.instrument()
  'new_instrument'

Developers and readers are encouraged to review the appropriate documents relating
to defining and writing an *astrodata* class for their new instrument's data. The
`Astrodata User's Manual` and "cheat sheet" were enumerated  earlier in
:ref:`Sec. 2.2, Related Documents <related>`.

