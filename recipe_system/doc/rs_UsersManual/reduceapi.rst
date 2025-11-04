.. reduceapi.rst

.. _reduceapi:

********************
The ``Reduce`` Class
********************

The ``Reduce`` class provides the underlying structure of the ``reduce``
command.  This section describes and discusses the programmatic interface
available on the class ``Reduce``.  This section is for users wanting to use
the ``Reduce`` class programmatically.

The ``reduce`` application introduced in the previous chapter is a user
interface script providing a command line access to the ``Reduce`` class.
The ``reduce`` application parses the arguments and initialize the ``Reduce``
class and its ``runr`` method.  It is possible to bypass the ``reduce``
command and sets attributes directly on an instance of ``Reduce``, as the
following discussion illustrates.

Using ``Reduce``
================
The Reduce class is defined in the ``recipe_system.reduction.coreReduce``
module.  The Reduce class provides a set of attributes and one public method,
``runr`` that launches a reduction. This is the only public method on the
class.

Very Basic Usage
----------------
The most basic usage involves importing the class, instantiating it, assigning
a file to reduce and then launching the ``runr`` method.

::

    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> myreduce = Reduce()
    >>> myreduce.files.append('S20161025S0111.fits')
    >>> myreduce.runr()

Typical Usage for Reduction
---------------------------
A more typical usage for reducing data can involve setting other options and
can include setting up a *logger*.  When using the Gemini data reduction
primitives, the logger is highly recommended.

Normal usage will also likely involve the use of the calibration database
facility, ``caldb``. We will ignore ``caldb`` here and rather fully describe
it and its usage in a subsequent chapter, :ref:`caldb`.  See :ref:`api_example`
where we put it all together.

::

    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> from gempy.utils import logutils
    >>>
    >>> logutils.config(file_name='example.log')
    >>>
    >>> inputfiles = ['S20161025S0200.fits', 'S20161025S0201.fits']
    >>> myreduce = Reduce()
    >>> myreduce.files = inputfiles
    >>> myreduce.runr()


Neither ``coreReduce`` nor the ``Reduce`` class initializes any logging activity.
This is the responsibility of the programmer.  The Recipe System does not
require a logger but the Gemini primitives do.  The absence of a logger when
using the Gemini data reduction package leads to double the reporting on
the screen.  More an annoyance than a problem, admittedly.

You are free to provide your own logger, or you can use the fully defined
logger provided in  DRAGONS. The Gemini primitives use some extra log levels in
addition to the standard ones. If you provide your own logger and plan to call
the Gemini primitives, the ``logutils.customize_logger()`` function in
``gempy.utils`` can be called to add these extra levels to your logger. This is a
lighter weight alternative to the ``logutils.config()`` function, which will
also replace any existing Handlers on the logger with the DRAGONS default ones.

Returning to the example above, we could also set the recipe to a custom
recipe, override a primitive parameters, set a data reduction package, etc.
The attributes that can be set are discussed in
:ref:`reduce_attributes` below.

::

    >>> myreduce.recipename = 'myrecipelib.myrecipe'
    >>> myreduce.uparms = dict([('stackFrames:operation', 'median')])
    >>> myreduce.dkpkg = 'thirdpartydr'
    >>> # rerun with the modified recipe and parameter
    >>> myreduce.runr()

A notable quirk is how to set the ``adpkg`` that is defined in the ``reduce``
command line interface.  The ``Reduce`` class does not have an attribute for
it.  Instead, the programmer must import any third party AstroData instrument
configuration files explicitely *before* launching ``runr``.

::

    >>> import astrodata
    >>> import thirdparty_instruments
    >>>
    >>> myreduce.Reduce()
    >>> myreduce.drpkg = 'thirdpartydr'
    >>> myreduce.files.append('filename.fits')
    >>> myreduce.runr()


.. _reduce_attributes:

Public Attributes to ``Reduce``
===============================

================    ========================   =======
Public Attribute    Python type                Default
================    ========================   =======
files               <type 'list' of 'str'>     []
output_filenames    <type 'list' of 'str'>     None
mode                <type 'str'>               'sq'
recipename          <type 'str'>               '_default'
drpkg               <type 'str'>               'geminidr'
suffix              <type 'str'>               None
ucals               <type 'dict'>              None
uparms              <type 'dict>               None
upload              <type 'list' of 'str'>     None
================    ========================   =======

**files**
    A list of input file names to reduce.  Only the first file in the list will
    be used for the recipe and primitive selection.

    ``myreduce.files.extend(['S20161025S0200.fits', 'S20161025S0201.fits'])``

**output_filenames**
    A list of output file names.  This **cannot** be set.  It is a return
    value. It is used *after* the recipe has run to collect the names of the
    files that were created.

    ``output_stack = myreduce.output_filenames[0]``

**mode**
    The reduction mode.  The Gemini data reduction package currently supports
    'sq' and 'qa', with 'ql' in the works. ['sq': Science Quality,
    'qa': Quality Assessment, 'ql': Quick Look Reduction.]

    ``myreduce.mode = 'qa'``

**recipename**
    The name of the recipe to use.  If left to "_default", the Recipe System
    will invoke the mappers and select the best matching recipe library and
    use its default recipe.

    If only the name of a recipe is provided, the
    mappers will be invoked to find the best matching recipe library and use
    the named recipe rather than the default.

    If a "module.recipe" string is provided, the user's "module" will be
    imported and the user's "recipe" will be used.  No mapping will be done.

    ``myreduce.recipename = 'myrecipelib.myrecipe'``

    If the name of a primitive is given, the Recipe System will find the best
    match primitive set and run the specified primitives from that set.


**suffix**
    The suffix to add the final outputs of a recipe.  In the Gemini primitives,
    default suffixes are assigned to each primitives.  Setting ``suffix``
    will override the default suffix of the last primitive in the recipe.

    ``myreduce.suffix = '_flatBfilter'``

**drpkg**
    The name of the data reduction package to use.  The default is ``geminidr``.
    If using a third-party package, or during new instrument development,
    set this attributes to import the correct suite of recipes and primitives.

    ``myreduce.drpkg = 'scorpiodr'``

**ucals**
    Set the processed calibration to be used.  This overrides the automatic
    selection done by the calibration manager, if one is being used.  This
    setting must be used if no calibration manager is used or available, or
    when, for example, the calibrations association rules are not yet
    implemented.  It is also useful for testing and for getting full control
    of the calibrations being used.

    The format for this attribute's value needs to looks like this::

        {'processed_bias': '/path/master_bias.fits'}

    There must be one entry for each type of calibrations.

    The recognized calibration types are currently:

    * processed_arc
    * processed_bias
    * processed_dark
    * processed_flat
    * processed_fringe
    * processed_slitillum
    * processed_standard

    The ``normalize_ucals`` function in the
    ``recipe_system.utils.reduce_utils`` module will convert a list of
    strings (in the format required for the command-line interface) into the
    appropriate dictionary *and* check that the specified calibrations files
    are readable and have the required tags. However, it is not necessary to
    make use of this function.

    Here's how to use ``normalize_ucals``::

        from recipe_system.utils.reduce_utils import normalize_ucals

        mycalibrations = ['processed_bias:/path/master_bias.fits',
                          'processed_flat:/path/master_Bflat.fits']

        myreduce.ucals = normalize_ucals(mycalibrations)


**uparms**
    Set primitive parameter values.  This will override the primitive
    defaults.  This is a dict with primitive name and parameter as the key,
    and the parameter value as the dict entry value. It can be built from
    a list of tuples with the primitive name and parameter
    in the first element, and the value in the second one.

    ``myreduce.uparms = dict([('stackFrames:operation', 'median')])``

    If the primitive name is omitted all parameters with that name, in any
    primitives will be reset.  Be careful.

**upload**
    **Internal use only**.  Specify which types of product to upload to the
    Gemini internal database.  Allowed values are "metrics", "calibs", and
    "science".

Using ``Reduce`` in a python script
-----------------------------------
If you use ``Reduce()`` in your own python script, it is imperative to make
use of the ``if __name__ == "__main__":`` python idiom in your script.

DRAGONS uses SExtractor for its source-finding and, since that's an external
package, python spawns a subprocess to run it. Unfortunately, the subprocess
it creates is a copy of the parent process and includes copies of all the
python objects in that process. In a normal reduction, this means copies of all
the images you’re reducing and so the memory footprint doubles in size and this
can be a problem because the reduction is already quite memory-intensive. We
discovered this the hard way in the early development stages of DRAGONS when
code would crash with out of memory errors.

Our solution is to create a subprocess as soon as the reduction starts and use
this subprocess to spawn SExtractor. By creating that subprocess right at the
start, it hasn't opened any files or done anything and so when it gets copied
to run SExtractor, the additional memory usage is small. But, the way that this
subprocess is created involves (effectively) re-importing all the python code
from the parent and, in your case, the "parent" is your script.

When python imports a module (such as your script) it runs the code, and so
another instance of your script will start running in the subprocess!
The solution is to "protect" the reduction part of your code in a way such that
it isn't run when imported, and only the preamble (import statements and the
like) is run. This is achieved by means of the python "idiom"
``if __name__ == "__main__"`` which is described in more detail
`here <https://realpython.com/if-name-main-python/>`_
(there are many others on the web) but basically means "only execute the
following code if I am the parent".