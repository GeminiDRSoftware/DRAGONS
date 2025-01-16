.. howto.rst
.. include discuss
.. include supptools

.. _howto:

How to Use It
=============

Introduction
------------

The ``reduce`` command is the DRAGONS Recipe System command line interface.
The Recipe System also provides an application programming interface (API),
whereby users and developers can programmatically invoke ``Reduce`` and set
parameters on an instance of that class.

Both interfaces allow users to configure and launch a Recipe System processing
pipeline on one or more input datasets. Control of the Recipe System
on the ``reduce`` command line is provided by a variety of options and switches.
All options and switches can also be accessed and controlled through the API.

This chapter will first present details of the ``reduce`` command line
interface, including an extended discussion of ???KL :ref:`atfile`. This
is followed by a detailed presentation on the Recipe System's ``Reduce``
class and it's :ref:`api`.

The ``reduce`` command
----------------------

Nominal Usage
+++++++++++++
We begin with the example shown in the :ref:`Introduction <intro>`::

  $ reduce S20161025S0111.fits

With no command line arguments or other options, a default mode of `'sq'`
is set. Instrument packages provide a default recipe for all instruments when
none is specified by the user. Under DRAGONS, all instrument packages define one
default recipe for each recipe library.

*Unless* passed an explicit recipe (-r, --recipename) and/or mode flag
(i.e. --qa or --ql), the Recipe System uses the dataset's astrodata `tags` attribute
and the Recipe System default mode `sq` to locate the appropriate recipe to run.

Within the ``DRAGONS`` package, `sq` recipe libraries for a dataset taken
with GMOS are defined in the ``geminidr`` package under::

  geminidr/
        gmos/
          recipes/
	        sq/
		   recipes_BIAS.py
		   recipes_FLAT_IMAGE.py
		   recipes_IMAGE.py


As previously indicated, the ``reduce`` command itself is deceptively simple
considering the processing that ensues. This simplicity is outward facing, which
means the complexity is "under the hood," as ``reduce`` and the ``Reduce`` class
use the ``astrodata`` abstraction to determine the recipes and primitive classes
appropriate to the dataset(s) presented.

Command Options and Switches
++++++++++++++++++++++++++++

The ``reduce`` command help is provided by the ``--help`` option. This help is
also available as a manual page as (``man reduce``). Subsequently, further
description and discussion of certain non-trivial options is presented. ::

  $ reduce --help
  usage: reduce [-h] [-v] [-d] [--adpkg ADPKG] [--drpkg DRPKG]
                [--logfile LOGFILE] [--logmode LOGMODE]
		[-p USERPARAM [USERPARAM ...]] [--qa] [--ql] [-r RECIPENAME]
		[--suffix SUFFIX] [--upload UPLOAD]
		[--user_cal USER_CAL [USER_CAL ...]]
		fitsfile [fitsfile ...]

  _____________________________ Gemini Observatory ____________________________
  ________________ DRAGONS Recipe Processing Management System ________________
  ______________________ Recipe System Release2.0 (beta) ______________________

  positional arguments:
    fitsfile              fitsfile [fitsfile ...]

 optional arguments:
  -h, --help            Show this help message and exit.
  -v, --version         Show program's version number and exit.
  -d , --displayflags   Display all parsed option flags and exit.
  --adpkg ADPKG         Specify an external astrodata definitions package.
                        This is only passed for non-Gemini instruments.The
                        package must be importable.
			E.g., --adpkg soar_instruments
  --drpkg DRPKG         Specify another data reduction (dr) package. The
                        package must be importable either through sys.path or
                        a user's PYTHONPATH. Recipe System default is 'geminidr'.
                        E.g., --drpkg ghostdr.
  --logfile LOGFILE     Set name of log file (default is 'reduce.log').
  --logmode LOGMODE     Set log mode: 'standard', 'quiet', 'debug'.
  -p USERPARAM [USERPARAM ...], --param USERPARAM [USERPARAM ...]
                        Set a parameter from the command line. The form '-p
                        par=val' sets a parameter such that all primitives
                        with that defined parameter will 'see' it. The form:
                        '-p primitivename:par=val', sets the parameter only
                        for 'primitivename'. Separate par/val pairs by
                        whitespace: (eg. '-p par1=val1 par2=val2').
  --qa                  Use 'qa' recipes. Default is to use 'sq' recipes.
  --ql                  Use 'quicklook' recipes. Default is to use 'sq' recipes.
  -r RECIPENAME, --recipe RECIPENAME
                        Specify a recipe by name. Users can request non-
                        default system recipe functions by their simple names,
                        e.g., -r qaStack, can request an explicit primitive
			function name, OR their own recipe file and recipe
			function. A user defined recipe function must be
			'dotted' with the recipe file. E.g.,
			'-r /path/to/recipes/recipefile.recipe_function'.
			For a recipe file in the current working directory,
                        only the file name is needed, as in, '-r
                        recipefile.recipe_function' The fact that the recipe
                        function is dotted with the recipe file name implies
                        that multiple user defined recipe functions can be
                        defined in a single file.
  --suffix SUFFIX       Add 'suffix' to filenames at end of reduction; strip
                        all other suffixes marked by '_'.
  --upload UPLOAD       Send these pipeline products to fitsstore. Default is
                        None. Eg., --upload metrics calibs science
  --user_cal USER_CAL   Specify user supplied calibrations for calibration
                        types. Eg., --user_cal gsTest_arc.fits .

These options are described in the following sections.

Informational switches
++++++++++++++++++++++
**-h, --help**
    show the help message and exit

**-v, --version**
    show program's version number and exit

**-d, --displayflags**
    Display all parsed option flags and exit.

    When specified, this switch presents a table of all parsed arguments and then
    exits. The table provides a convenient view of all passed and default values.
    When not specified, 'recipename' indicates 'None' because at this point the
    Recipe System has not been invoked and a default recipe not yet determined.
    Eg.,::

       $ reduce -d --logmode quiet fitsfile.fits

	  Literals			var 'dest'		Value
	 -----------------------------------------------------------------
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
	['--logfile']                   :: logfile              :: reduce.log
	-----------------------------------------------------------------

       Input fits file(s):	fitsfile.fits

.. _options:

Configuration Switches, Options
+++++++++++++++++++++++++++++++
**--adpkg <ADPKG>**
    Specify an external astrodata definitions package. This is only passed for
    non-Gemini instruments.The package must be importable.
    E.g., --adpkg soar_instruments

**--logfile <LOGFILE>**
    Set the log file name. Default is 'reduce.log' in the current directory.

**--logmode <LOGMODE>**
    Set logging mode. One of

    * standard
    * quiet
    * debug

    'quiet' writes only to the log file. Default is 'standard'.

**--drpkg DRPKG**
    Specify an external data reduction (dr) package. The package must be
    importable. Default is 'geminidr'.

    E.g., ``--drpkg ghostdr``

    When this option is specified, users will see the passed value for
    'drpkg'using the [-d --displayflags] option. We shall also include the
    --adpkg option for Ghost data. For the example above::

     $ reduce -d --adpkg ghost_instruments --drpkg ghostdr --logmode quiet --qa
       -r display S20150929S0151.fits

        --------------------   switches, vars, vals  --------------------

	  Literals			var 'dest'		Value
	 -----------------------------------------------------------------
	['-d', '--displayflags']        :: displayflags         :: True
	['-p', '--param']               :: userparam            :: None
	['--logmode']                   :: logmode              :: quiet
	['--ql']                        :: mode                 :: qa
	['--qa']                        :: mode                 :: qa
	['--upload']                    :: upload               :: None
	['-r', '--recipe']              :: recipename           :: display
	['--adpkg']                     :: adpkg                :: ghost_instruments
	['--suffix']                    :: suffix               :: None
	['--drpkg']                     :: drpkg                :: ghostdr
	['--user_cal']                  :: user_cal             :: None
	['--logfile']                   :: logfile              :: reduce.log
	-----------------------------------------------------------------

     Input fits file(s):	S20150929S0151.fits

**-p <USERPARAM [USERPARAM ...]>, --param <USERPARAM [USERPARAM ...]>**
    Set a primitive parameter from the command line. The form ``-p par=val`` sets
    the parameter such that all primitives will 'see' it. The form

    ``-p primitivename:par=val``

    sets the parameter such that it applies only when the primitive is
    'primitivename'. Separate parameter-value pairs by whitespace:
    (eg. '-p par1=val1 par2=val2')

    See :ref:`userpars`, for more information on these values.

**--qa**
    Set the ``mode`` attribute to 'qa'. Default is 'sq'. Note: there is no
    ``--mode`` option. ``mode`` is an attribute on the Reduce class which is
    set by the this flag and/or the following ``--ql`` flag. See the reduce
    example table above.

**--ql**
    Set the ``mode`` attribute to 'ql'. Default is 'sq'. Note: there is no
    flag, ``--mode``. ``mode`` is an attribute on the Reduce class which is
    set by the this flag and/or the previous ``--qa`` flag. See the reduce
    example table above.

**-r <RECIPENAME>, --recipe <RECIPENAME>**
    Specify a recipe by name. Users can request non-default system recipe
    functions by their simple names, e.g., ``-r stack``, OR may specify
    their own recipe file and recipe function. A user defined recipe function
    must be 'dotted' with the recipe file.

    E.g.
    ::

      -r /path/to/recipes/recipefile.recipe_function

    For a recipe file in the current working directory (cwd), only the file name
    is needed
    ::

     -r recipefile.recipe_function

    The fact that the recipe function is dotted with the recipe file name implies
    that multiple user recipe functions can be defined in a single file, i.e.
    a recipe library.

    Readers should understand that these recipe files must be *python modules*
    and named accordingly. That is, in the example above, 'recipefile' is a
    python module named, ``'recipefile.py'``

    Finally, the specified recipe can be an *actual primitive function name*::

      -r display

    and the Recipe System will display the dataset in an open and available
    viewer, such as ds9.

**--suffix <SUFFIX>**
    Add 'suffix' to output filenames at end of reduction.

**--upload**
    Send the following pipeline products to fitsstore. Default is None.
    E.g.::

      --upload metrics calibs

    OR equivalently::

      --upload=metrics,calibs

**--user_cal <USER_CAL [USER_CAL ...]>**
    The option allows users to provide their own calibrations to ``reduce``.
    Add a calibration to User Calibration Service. The user calibration must include
    the calibration type. *Only* processed calibrations should be specified::

     --user_cal processed_arc:wcal/gsTest_arc.fits

.. _userpars:

Overriding Primitive Parameters
+++++++++++++++++++++++++++++++

In some cases, users may wish to change the functional behaviour of certain
processing steps, such as changing default parameters of primitive functions.

Each primitive has a set of system-defined parameters, which are used to control
functional behaviour of the primitive. Users can adjust parameter values from the
reduce command line with the option,

    **-p, --param**

Parameters and values specified through the **-p, --param** option will `override`
the parameter default value and may alter default behaviour of the
primitive accessing this parameter. A user may pass several parameter-value pairs
with this option.

Eg.::

  $ reduce -p operation=mean nhigh=4 nlow=2 S20161025S0111.fits

User-specified parameter values can be focused on one primitive. For example,
if a parameter applies to more than one primitive, like ``operation``, you can
explicitly direct a new parameter value to a particular primitive. The 'detection
threshold' has a defined default, but a user may alter this parameter default to
change the source detection behaviour::

 $ reduce -p stackFlats:operation=mean nhigh=4 nlow=2 S20161025S0111.fits

How is this command line parsed? The ``operation`` parameter for the ``stackFlats``
primitive function is set to ``mean``. All other primitives having an "operation"
parameter are unaffected, while the ``nhigh`` and ``nlow`` parameters remain
unqualified and applicable to all primitive parameters with the same name.

Because of the complex hierarchy of the geminidr primitive classes and their
associated parameter classes, DRAGONS provides the ``showpars`` command line tool
that allows users to view available parameters for a given dataset and primitive
function. For further information and instruction on how to use ``showpars`` to
display settable primitive parameters, see
:ref:`Supplemental Tools, Sec 4.1 <showpars>`.

.. _atfile:

The @file facility
------------------

The reduce command line interface supports what might be called an 'at-file'
facility (users and readers familiar with IRAF will recognize this facility).
An `@file` allows users to provide any and all command line options and flags
to ``reduce`` in an acsii text file. The example command in the previous section
can be written into a file, in whole or in part. Here, we write the desired
parameters to a file called ``reduce_args.par``::

  -p
  stackFlats:operation=mean
  nhigh=4
  nlow=2

And now the ``reduce`` command looks like, ::

  $ reduce @reduce_args.par S20161025S0111.fits

By passing an `@file` to ``reduce`` on the command line, users can encapsulate
all the options and positional arguments they may wish to specify in a single
`@file`. It is possible to use multiple `@file` s and even to embed one or more
`@file` s in another. The parser opens all files sequentially and parses
all arguments in the same manner as if they were specified on the command line.
Essentially, an `@file` is some or all of the command line and parsed identically.

To further illustrate the convenience provided by an `@file`, we'll continue
with an example `reduce` command line that has even more arguments. We will
also include new positional arguments, i.e., file names::

  $ reduce -p stackFlats:operation=mean nhigh=4 nlow=2
    -r recipe.ArgsTest S20130616S0019.fits N20100311S0090.fits

Ungainly, to be sure. Here, three (3) `user parameters` are being specified
with **-p**, a `recipe` with **-r**. We can write these parameters into our
plain text `@file` called `reduce_args.par`::

   S20130616S0019.fits
   N20100311S0090.fits
   --param
   stackFlats:operation=mean
   nhigh=4
   nlow=2
   -r recipe.ArgsTests

This then turns the previous reduce command line into something a little more
`keyboard friendly`::

  $ reduce @reduce_args.par

The order of arguments in an `@file` is irrelevant, as is the file's name. The above
file could present the arguments in completely different orders and forms, such as::

  -r recipe.ArgsTests
  --param
  stackFlats:operation=mean
  nhigh=4 nlow=2
  S20130616S0019.fits
  N20100311S0090.fits

Readers will note the two parameters, nhigh, nlow, written on the same line in the
above example. This is perfectly fine and just as you would have it on the command
line. All white space is equivalent to the command line parser. The parser sees no
difference across white space characters, such as space, tab, newline, etc..

Comments are accommodated, both full line and in-line with the ``#``
character. Because all white space is treated identically, the user can
choose to "arrange" their `@file` for clarity.

Here's a more readable version of the example file using comments and tabulation::

    # Gemini Observatory
    # DRAGONS
    # reduce parameter file

    # Spec the recipe
    -r
        recipe.ArgsTests         # test recipe

    # primitive parameters here
    --param
        stackFlats:operation=mean
	nhigh=4
	nlow=2

    S20130616S0019.fits
    N20100311S0090.fits

All these example of the ``reduce_args.par`` are parsed equivalently, which users
may confirm by adding the **-d** flag::

  $ reduce -d @reduce_args.par

  --------------------   switches, vars, vals  --------------------

  Literals			var 'dest'		Value
  -----------------------------------------------------------------
  ['-d', '--displayflags']      :: displayflags     :: True
  ['-p', '--param']             :: userparam        :: ['stackFlats:operation=mean',
                                                       'nhigh=4','nlow=2']
  ['--logmode']                 :: logmode          :: standard
  ['--ql']                      :: mode             :: sq
  ['--qa']                      :: mode             :: sq
  ['--upload']                  :: upload           :: None
  ['-r', '--recipe']            :: recipename       :: recipe.ArgsTests
  ['--adpkg']                   :: adpkg            :: None
  ['--suffix']                  :: suffix           :: None
  ['--drpkg']                   :: drpkg            :: geminidr
  ['--user_cal']                :: user_cal         :: None
  ['--logfile']                 :: logfile          :: reduce.log
  -----------------------------------------------------------------

  Input fits file(s):	S20130616S0019.fits
  Input fits file(s):	N20100311S0090.fits

Recursive @file processing
++++++++++++++++++++++++++

As implemented, the `@file` facility will recursively handle, and process
correctly, other `@file` specifications that appear in a passed `@file` or
on the command line. For example, we may have another file containing a
list of fits files, separating ``reduce`` options from positional
arguments.

We have a plain text 'fitsfiles' file containing the line::

  test_data/S20130616S0019.fits

We can indicate that this file is to be consumed with the prefix character
"@" as well::

  # reduce test parameter file

  @fitsfiles             # file with fits files

  # primitive parameters.
  --param
  stackFlats:operation=mean
  nhigh=4
  nlow=2

  # Spec the recipe
  -r recipe.ArgTests

The parser will open and read the @fitsfiles, consuming those lines in the
same way as any other command line arguments. Indeed, such a file need not only
contain fits files (positional arguments), but other arguments as well. This is
recursive. That is, the @fitsfiles can contain other "at-files", which can contain
other "at-files", which can contain ..., etc. These will be processed
serially.

Continuing the example, we'll name this `@file`  ``parfile``.

As stipulated earlier, because the `@file` facility provides arguments equivalent
to those that appear on the command line, employment of this facility means that
a reduce command line could assume the form::

   $ reduce @parfile @fitsfiles

or equivalently::

   $ reduce @fitsfiles @parfile

where 'parfile' might contain the flags and user parameters, and 'fitsfiles'
could contain a list of datasets.

Eg., fitsfiles comprises the one line::

  test_data/N20100311S0090.fits

while parfile holds all other specifications::

  # reduce test parameter file
  # GDPSG

  # primitive parameters.
  --param
    stackFlats:operation=mean
    nhigh=4
    nlow=2

  # Spec the recipe
  -r recipe.ArgTests

The `@file` does not need to be located in the current directory.  Normal shell
syntax applies, for example::

   reduce @../../parfile @fitsfile

Overriding @file values
+++++++++++++++++++++++
The ``reduce`` application employs a customized command line parser such that
the command line option

**-p** or **--param**

will accumulate a set of parameters `or` override a particular parameter.
This may be seen when a parameter is specified in a user `@file` and then
specified on the command line. For unitary value arguments, the command line
value will `override` the `@file` value.

It is further specified that if one or more datasets (i.e. positional arguments)
are passed on the command line, `all fits files appearing as positional arguments
in the parameter file will be replaced by the command line arguments.`

Using the parfile above,

Eg. 1)  Accumulate a new parameter::

  $ reduce @parfile --param FOO=BARSOOM

  parsed options:
  ---------------
  FITS files:    ['S20130616S0019.fits', 'N20100311S0090.fits']
  Parameters:    stackFlats:operation=mean, nhigh=4, nlow=2, FOO=BARSOOM
  RECIPE:        recipe.ArgsTest

Eg. 2) Override a parameter in the `@file`::

  $ reduce @parfile --param nhigh=5

  parsed options:
  ---------------
  FITS files:    ['S20130616S0019.fits', 'N20100311S0090.fits']
  Parameters:    stackFlats:operation=mean, nhigh=5, nlow=2
  RECIPE:        recipe.ArgsTest

Eg. 3) Override the recipe::

  $ reduce @parfile -r recipe.FOO

  parsed options:
  ---------------
  FITS files:    ['S20130616S0019.fits', 'N20100311S0090.fits']
  Parameters:    stackFlats:operation=mean, nhigh=4, nlow=2
  RECIPE:        recipe.FOO

Eg. 4) Override a recipe and specify another fits file. The file names in
the `@file` will be ignored::

  $ reduce @parfile -r recipe.FOO test_data/N20100311S0090_1.fits

  parsed options:
  ---------------
  FITS files:    ['test_data/N20100311S0090_1.fits']
  Parameters:    stackFlats:operation=mean, nhigh=4, nlow=2
  RECIPE:        recipe.FOO

.. _api:

Application Programming Interface (API)
---------------------------------------
The ``Reduce`` class provide the underlying structure of the ``reduce`` command.
This section describes and discusses the programmatic interface available on
the class ``Reduce``.  This section is for advanced users wanting to use the
``Reduce`` class programmatically.

The ``reduce`` application is essentially a skeleton script providing the
described command line interface. After parsing the command line, the script
then passes the parsed arguments to its main() function, which in turn calls
the ``Reduce()`` class constructor with the command line "args". Programmatically,
one bypasses the ``reduce`` command and sets attributes directly on an instance
of ``Reduce``, as the following discussion illustrates.

Class Reduce, the runr() method, and logging
++++++++++++++++++++++++++++++++++++++++++++

The Reduce class is defined under ``DRAGONS`` in the ``recipe_system.reduction``
module, ``coreReduce.py``.

The Reduce class is importable and provides settable attributes and a callable
that can be used programmatically. Callers need not supply an "args" parameter
to the class initializer, i.e. __init__(). An instance of Reduce will have all
the same arguments as in a command line scenario, available as attributes on the
instance. Once an instance of Reduce is instantiated and instance attributes
set as needed, there is one public method to call, **runr()**. This is the only
public method on the class.

E.g.,

>>> from recipe_system.reduction.coreReduce import Reduce
>>> myreduce = Reduce()
>>> myreduce.files
[]
>>> myreduce.files.append('S20130616S0019.fits')
>>> myreduce.files
['S20130616S0019.fits']

Or callers may simply set the ``files`` attribute to be an existing list of files

>>> fits_list = ['FOO.fits', 'BAR.fits']
>>> myreduce.files = fits_list

On the command line, you can specify a recipe with the ``-r`` [ ``--recipe`` ]
flag. Programmatically, callers set the recipe directly::

>>> myreduce.recipename = 'recipe.MyRecipe'

All other properties and  attributes on the API may be set in standard pythonic
ways. See Appendix :ref:`Class Reduce: Settable properties and attributes <props>`
for further discussion and more examples.

Neither ``coreReduce`` nor the Reduce class initializes any logging activity. This
is the responsibility of outside parties, as in the case of the ``reduce`` script,
which configures the logging facility before any processing begins. Should you wish
to log the processing steps -- probably true -- you will have to initialize your
own "logger". You are free to provide your own logger, or you can use the fully
defined logger provided in  DRAGONS. It is recommended that you use this system
logger, as the ``reduce`` command line options, and corresponding Reduce attributes,
are tuned to use the DRAGONS logger. You will see logger configuration calls in
the examples below. For details on how to configure the DRAGONS logger, see
:ref:`Using the logger <logger>`.


Call the runr() method
^^^^^^^^^^^^^^^^^^^^^^

Once you are satisfied that all attributes are set to the desired values, and
the logger is configured, the runr() method on the "reduce" instance may then be
called. The following brings the examples above into one "end-to-end" use of
Reduce and logutils::

  >>> from recipe_system.reduction.coreReduce import Reduce
  >>> from gempy.utils import logutils
  >>> logutils.config(file_name='my_reduce_run.log')
  >>> reduce = Reduce()
  >>> reduce.files.append('S20130616S0019.fits')
  >>> reduce.recipename = 'recipe.MyRecipe'
  >>> reduce.runr()

  All submitted files appear valid
  ================================================================================
  RECIPE: recipe.MyRecipe
  ================================================================================
  ...

Processing will then proceed in the usual manner. Readers will note that
callers need not create more than one Reduce instance in order to call runr()
with a different dataset or options.

Eg.,::

 >>> from recipe_system.reduction.coreReduce import Reduce
 >>> from gempy.utils import logutils
 >>> logutils.config(file_name='my_reduce_run.log')
 >>> reduce = Reduce()
 >>> reduce.files.append('S20130616S0019.fits')
 >>> reduce.recipename = 'recipe.MyRecipe'
 >>> reduce.runr()
   ...
 reduce completed successfully.

 >>> reduce.recipename = 'recipe.NewRecipe'
 >>> reduce.files = ['newfile.fits']
 >>> reduce.userparam = ['nhigh=5']
 >>> reduce.runr()

Once an attribute is set on an instance, such as above with ``userparam``, it is
always set on the instance. If, on another call of runr() the caller does not
wish to change ``nhigh``, simply reset the attribute::

>>> reduce.userparam = []
>>> reduce.runr()

Readers may wish to review the examples in Appendix
:ref:`Class Reduce: Settable properties and attributes <props>`

.. _logger:

Using the logger
^^^^^^^^^^^^^^^^

.. note:: When using an instance of Reduce() directly, callers must configure
	  their own logger. Reduce() does not configure logutils prior to using
	  a logger as returned by logutils.get_logger(). The following discussion
	  demonstrates how this is easily done. It is `highly recommended`
	  that callers configure the logger.

It is recommended that callers of Reduce use a logger supplied by the DRAGONS
module ``logutils``. This module employs the python logger module, but with
recipe system specific features and embellishments. The recipe system and pipelines
defined within DRAGONS will expect to have access to a logutils logger object,
which callers should provide prior to calling the ``runr()`` method.

To use ``logutils``, import, configure, and get::

  from gempy.utils import logutils
  logutils.config(file_name="test.log", mode="standard")
  log = logutils.get_logger(__name__)

where ``__name__`` is usually the calling module's __name__ property, but can
be any string value. Once configured and instantiated, the ``log`` object is
ready to use. See section :ref:`options` for logging modes described on the
``--logmode`` option.

The ``reduce`` command line provides default values for the configuration of the
logger as described in Sec. :ref:`options`. Users may adjust these values and then
pass them to the ``logutils.config()`` function, or pass other values directly
to ``config()``, as shown above. This is precisely what ``reduce`` does when it
configures logutils. See Sec. :ref:`options`  and
Appendix :ref:`Class Reduce: Settable properties and attributes <props>` for
allowable and default values of these and other options.

.. note:: logutils.config() may be called mutliply, should callers
	  want to change logfile names for different calls on runr().
