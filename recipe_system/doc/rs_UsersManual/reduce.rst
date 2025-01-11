.. reduce.rst

.. _reduce:

**********************
The ``reduce`` command
**********************

Introduction
============
The ``reduce`` command is the DRAGONS Recipe System command line interface.
The Recipe System also provides an application programming interface (API),
whereby users and developers can programmatically invoke ``Reduce`` and set
parameters on an instance of that class (see :ref:`reduceapi`).

Both interfaces allow users to configure and launch a Recipe System processing
pipeline on one or more similar input datasets.  Control of the Recipe System
on the ``reduce`` command line is provided by a variety of options and
switches which we will introduce in this chapter.

.. todo: say that the caldb is a vital companion of reduce and refer to the
         chapter dedicated to it.

Usage Examples
==============
Below we show examples that a user might typically want to do when using
``reduce``.  The command offers a lot of flexibility though, these examples
are just a small subset of the possibilities.  The objective here is to help
the user get started.

Nominal usage
-------------
Because the Recipe System is automated, in many cases all that is needed is
the command and a filename.

::

    reduce S20161025S0111.fits

The system defaults to the "sq" mode, ie. science quality recipes.  The best
match recipe will be used with the best match primitive set.  The required
processed calibrations will be fetched from the :ref:`caldb`.

The system defaults to using the Gemini Astrodata configuration package and
the Gemini data reduction package, ``gemini_instruments`` and ``geminidr``,
respectively.


Overriding Primitive Parameters
-------------------------------
The primitives in each set are given default values that have been found to
give good results in most cases.  Depending on the data and the science
objectives, it might be necessary to tweak the primitive parameters to
optimize the reduction.  The ``-p``, or in long form ``--param`` option allows
the user to override the defaults.

::

    reduce S20161025S0111.fits -p stackFrames:operation=median \
           stackFrames:reject_method=minmax

This sets the ``stackFrames`` input parameters ``operation`` and
``reject_method`` to ``median`` and ``minmax``, respectively.

As one can see that, if several parameters are to be modified, the command can
grow rather long.  There is a way to keep it clean, see the section below
on the :ref:`@file facility<atfile>`.


Calling Specific Recipes and Primitives
---------------------------------------
The Recipe System's default behavior is to select the best recipe
automatically.  It is however possible, and sometimes required, to override
this.

Override the default recipe
+++++++++++++++++++++++++++
The first case where the recipe selection can be overridden is to select a
recipe in the library different from the default.  A recipe library can
contain more than one recipe.  Only one is set as the default.  To let the
Recipe System select the most appropriate recipe library, but then request
the use of recipe within that library other than the default, simply state
the name of the desired recipe.  A good example is when making a bad pixel
mask (BPM) for NIRI::

    reduce @flats @darks -r makeProcessedBPM

Here the Recipes System will find the recipe library for NIRI flats (because
the flats are first in the list), and then instead of running the default
recipe which would in this case make a processed flat, it will run the
``makeProcessedBPM`` recipe.

For information about the ``@`` format, see :ref:`atfile` below.

User recipe
+++++++++++
It is possible for the user to force the use of a custom recipe.  This is
done with the ``-r`` flag again.  The structure "recipe library containing
recipes" must still be obeyed.  Here is how the request is made::

    reduce S20161025S0111.fits -r myrecipelibrary.myspecialrecipe

Both the name of the recipe library and, after the dot, the name of the
recipe function are required.  The path to the library can be prepended.

Calling a single primitive
++++++++++++++++++++++++++
Single primitives can be called directly from the command line bypassing the
recipes entirely.  A useful case is when one wants to display dataset.  There
is a primitive named ``display``.  The Recipe System will find the best-match
primitive set, and then run the ``display`` primitive it contains.

::

    reduce S20161025S0111.fits -r display


Manually Setting Calibrations
-----------------------------
When the calibration manager is not available or if working on a new type
of data not yet coded in the calibration association rules, it will be
necessary to specify the processed calibration to use on the command line.

Another situation would be if one wanted to try various version of a calibration
or different calibrations altogether to try to optimize a reduction.  In such
a case, one needs full control on which calibration is being used rather than
always using the "best-match" returned by the local calibration manager.

::

    reduce S20161025S0111.fits --user_cal processed_bias:S20161025S0200_bias.fits


Command Line Options and Switches
=================================
The ``reduce`` command help is provided by the ``--help`` option. This help is
also available as a manual page as (``man reduce``).  The options and switches
are described further here.

Information Switches
--------------------
**-h, --help**
    show the help message and exit

**-v, --version**
    show program's version number and exit

**-d, --displayflags**
    Display all parsed option flags and exit.

    The table provides a convenient view of all passed and default values
    for ``reduce``.  This can be useful when wanting to verify the syntax of
    a  ``reduce`` call and to make sure everything has been parsed as expected.

    Note that when not specified, `recipename` indicates 'None' because at
    this point in the execution the Recipe System has not yet been invoked and
    a default recipe not yet been determined.
    Eg.,

    ::

       $ reduce -d --logmode quiet fitsfile.fits

         Literals			var 'dest'		Value
         ---------------------------------------------------------------------
         ['-d', '--displayflags']        :: displayflags         :: True
         ['-p', '--param']               :: userparam            :: None
         ['--logmode']                   :: logmode              :: quiet
         ['--ql']                        :: mode                 :: sq
         ['--qa']                        :: mode                 :: sq
         ['--upload']                    :: upload               :: None
         ['-r', '--recipe']              :: recipename           :: None
         ['--adpkg']                     :: adpkg                :: None
         ['--suffix']                    :: suffix               :: None
         ['--drpkg']                     :: drpkg                :: geminidr
         ['--user_cal']                  :: user_cal             :: None
         ['-c', '--config']              :: config_file          :: None
         ['--logfile']                   :: logfile              :: reduce.log
         ---------------------------------------------------------------------

         Input fits file(s):	fitsfile.fits


Configuration Switches and Options
----------------------------------
**--adpkg <ADPKG>**
    Specify an external AstroData configuration package. This is used for
    non-Gemini instruments or during development of a new Gemini instrument.
    The package must be importable.  The default AstroData configuration
    package is ``gemini_instruments`` and it is distributed with DRAGONS.

    E.g., ``--adpkg scorpio_instruments``


**--drpkg DRPKG**
    Specify an external data reduction package. This is used for
    non-Gemini instruments or during development of a new Gemini instrument.
    The package must be importable. The default data reduction package is
    ``geminidr`` and it is distributed with DRAGONS.

    E.g., ``--drpkg scorpiodr``


**--logfile <LOGFILE>**
    Set the log file name. The default is ``reduce.log`` and it is written in
    the current directory.


**--logmode <LOGMODE>**
    Set logging mode. One of

    * standard
    * quiet
    * debug

    "quiet" writes only to the log file. The other modes writes information
    to the screen and to the log file.  The default is "standard".


**-p <USERPARAM [USERPARAM ...]>, --param <USERPARAM [USERPARAM ...]>**
    Set a primitive input parameter from the command line.  The form is

    ``-p primitivename:parametername=value``

    This sets the parameter such that it applies only for the primitive
    "primitivename". To set multiple parameter-value pairs, separate them with
    whitespace, eg. ``-p par1=val1 par2=val2``

    The form ``-p parametername=value`` is also allowed but beware, that will
    sets any parameter with that name from any primitives to that value. It
    is somewhat dangerous and of limited use.  It is to be seen as a global
    setting.


**--qa**
    Set the **mode** of operation to "qa", "quality assessment". When no "qa"
    or "ql" flag are specified the default mode is "sq".  The "qa" mode is use
    internally at Gemini.  Recipes differ depending on the mode.


**--ql**
    Set the **mode** of operation to "ql", "quicklook". When no "qa"  or "ql"
    flag are specified the default mode is "sq".  The "ql" mode is use for
    quick, near science quality reduction.  Science quality is not guaranteed.
    Recipes differ depending on the mode.


**-r <RECIPENAME>, --recipe <RECIPENAME>**
    Specify a recipe by name. Users can request a non-default system recipe
    by names, e.g., ``-r makeProcessedBPM``, or may specify their own recipe
    library and recipe function within.  A user-defined recipe function
    must be "dotted" with the recipe file.

    ::

      -r /path/to/recipes/recipelibrary.recipename

    For a recipe file in the current working directory, the path can be
    omitted::

     -r recipelibrary.recipename

    A recipe library can contain more than one recipe.  The recipe library
    must be a Python module, eg. ``recipelibrary.py``.  The recipes are
    Python functions within that module.

    Finally, instead of specifying a recipe, it is possible to specify a
    primitive::

      -r display


**--suffix <SUFFIX>**
    Add "suffix" to output filenames at the end of the reduction.


**--upload**
    **Currently used internally (Gemini) only.**

    Send specific pipeline products to internal database. The default is None.

    ::

      --upload metrics calibs

    or equivalently::

      --upload=metrics,calibs


**--user_cal <USER_CAL [USER_CAL ...]>**
    Specify which processed calibration to use for the reduction.  This
    override the selection from the local calibration manager.  The syntax is::

      --user_cal calibrationtype:path/calibrationfilename

    Eg.::

      --user_cal processed_bias:somepath/processed_bias.fits

    The recognized calibration types are currently:

    * processed_arc
    * processed_bias
    * processed_dark
    * processed_flat
    * processed_fringe
    * processed_slitillum
    * processed_standard

**-c <CONFIGFILE>, --config <CONFIGFILE>**
    Specify a configuration file for DRAGONS. By default, the file indicated
    by the ``$DRAGONSRC`` environment variable will be used or, if that
    variable is not defined, then the default ``~/.dragons/dragonsrc``. This
    switch will take priority to use the specified configuration file.


.. _atfile:

The @file Facility
==================
The reduce command line interface supports an "at-file" facility.
An ``@file`` allows users to provide any and all command line options and flags
to ``reduce`` in an acsii text file.  This tool is very useful to keep the
command line to a reasonable length and also to keep a record of the
configurations that are applied.  Here we illustrate how to use it.

Basic @file Usage
-----------------
In a previous section we had an example where we were modifying a primitive's
input parameter values.

::

    reduce S20161025S0111.fits -p stackFrames:operation=median \
           stackFrames:reject_method=minmax

Instead of typing the parameter settings on the command line, it might be
more convenient to use an "at-file".  We can write the parameter information
in the "at-file" and add it to our ``reduce`` call.  Let us have a file
named "myreduction.par" with this content::

    -p
    stackFrames:operation=median
    stackFrames:reject_method=minmax

Now we can call ``reduce`` as follow::

    reduce S20161025S0111.fits @myreduction.par

By passing an ``@file`` to ``reduce`` on the command line, users can encapsulate
all the options and positional arguments they may wish to specify in a single
``@file``. It is possible to use multiple ``@file`` and even to embed one or
more ``@file`` in another (see :ref:`recursive`). The parser opens all files
sequentially and parses all arguments in the same manner as if they were
specified on the command line.

To further illustrate the convenience provided by an ``@file``, we'll continue
with an example ``reduce`` command line that has even more arguments. We will
also include new positional arguments, i.e., file names::

  $ reduce -p stackFrames:operation=median stackFrames:reject_method=minmax \
    -r myrecipelib.myrecipe S20161025S0200.fits S20161025S0201.fits \
    S20161025S0202.fits S20161025S0203.fits S20161025S0204.fits

Here, two user parameters are being specified with ``-p``, a recipe with
``-r``, and a list of input datasets.  We can write all this into a plain text
``@file``, let's name it "reduce_args.par"::

    # input data files
    S20161025S0200.fits
    S20161025S0201.fits
    S20161025S0202.fits
    S20161025S0203.fits
    S20161025S0204.fits

    # primitive parameters optimization
    --param

        # stackFrames
        stackFrames:operation=median
        stackFrames:reject_method=minmax

    # recipe
    -r
        myrecipelib.myrecipe

Now we can call ``reduce`` this way::

    reduce @reduce_args.par

The order of the arguments in an ``@file`` is irrelevant, as is the file name.
Also, the parser sees no difference across white space characters, such as
space, tabs, newlines, etc.  Comments are accommodated, both full line and
in-line with the ``#``
character.

Finally, the "at-file" does not need to be in the current directory.  A path
can be given.  For example::

    reduce @../reduce_args.par



.. _recursive:

Recursive @file Usage
---------------------
As implemented, the ``@file`` facility will recursively handle and process
other ``@file`` specifications that appear in a ``@file`` or
on the command line. For example, we may have another file containing a
list of input files, let's call it "bias.lis"::

    # raw biases
    S20161025S0200.fits
    S20161025S0201.fits
    S20161025S0202.fits
    S20161025S0203.fits
    S20161025S0204.fits

Then, we can add this list as an "at-file" in the ``reduce_args.par`` file::

    # input files
    @bias.lis

    # primitive parameters optimization
    --param

        # stackFrames
        stackFrames:operation=median
        stackFrames:reject_method=minmax

    # recipe
    -r
        myrecipelib.myrecipe

The ``reduce`` call becomes::

    reduce @reduce_args.par

The parser will open and read the @bias.lis, consuming those lines in the
same way as any other command line arguments. Indeed, such a file need not only
contain fits files (positional arguments), but other arguments as well. This is
recursive. That is, the @fitsfiles can contain other "at-files", which can
contain other "at-files", which can contain ..., etc. These will be processed
serially.

Or one might want to keep the input files and the parameter settings separate.
Then if we remove the ``@bias.lis`` from the "reduce_args.par" files, we can
use it explicitly on the ``reduce`` command line::

    reduce @bias.lis @reduce_args.par



Overriding @file Values
-----------------------
The ``reduce`` application employs a customized command line parser such that
the command line option given in the ``@file`` can be modified on the command
line *after* the ``@file`` has been processed.


The ``-p`` or ``--param`` will accumulate a set of parameters *or* override a
particular parameter.  This may be seen when a parameter is specified in a
user ``@file`` and then specified on the command line.  See Example 1 and 2
below.

For unitary value arguments, the command line value will *override* the
``@file`` value.  See Example 3 below.

It is further specified that if one or more datasets (i.e. positional
arguments) are passed on the command line, **all** files appearing as
positional arguments in the "at-file" will be **replaced** by the one(s) on the
command line. See Example 4 below.

In all cases, remember to use the ``-d`` option to verify the parsing if you
are not sure.

Examples
++++++++

The ``@file`` used in the examples, "reducepar", contains::

    # input data files
    S20161025S0200.fits
    S20161025S0201.fits
    S20161025S0202.fits
    S20161025S0203.fits
    S20161025S0204.fits

    # primitive parameters optimization
    --param

        # stackFrames
        stackFrames:operation=median

    # recipe
    -r
        myrecipelib.myrecipe


**Example 1**:  Accumulate a new parameter::

    reduce @reducepar --param stackFrames:hsigma=5.0

    Summary of parsed options:
    --------------------------
    Input files: no changes
    Parameters: ['stackFrames:operation=median', 'stackFrames:hsigma=5.0']
    Recipe: no changes

**Example 2**: Override a parameter defined in the ``@file``::

    reduce @reducepar --param stackFrames:operation=wtmean

    Summary of parsed options:
    --------------------------
    Input files: no changes
    Parameters: ['stackFrames:operation=wtmean']
    Recipe: no changes


**Example 3**: Override the recipe::

    reduce @reducepar -r myrecipelib.different_recipe

    Summary of parsed options:
    --------------------------
    Input files: no changes
    Parameters: no changes
    Recipe: myrecipelib.different_recipe


**Example 4**: Override the input files.  All the files in the ``@files`` will
be ignored::

    reduce @reducepar S20161025S0111.fits

    Summary of parsed options:
    --------------------------
    Input files: S20161025S0111.fits
    Parameters: no changes
    Recipe: no changes
