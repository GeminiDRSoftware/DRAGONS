.. interfaces:

Interfaces
==========

Introduction
------------

The ``reduce`` application provides a command line interface and an API, both
of which can configure and launch a Recipe System processing pipeline (a 'recipe')
on the input dataset. Control of ``reduce`` and the Recipe System is provided 
by a variety of options and switches. Of course, all options and switches 
can be accessed and controlled through the API.


Command line interface
----------------------

We begin with the command line help provided by ``reduce --help``, followed by 
further description and discussion of certain non-trivial options that require 
detailed explanation. ::

  usage: reduce [options] fitsfile [fitsfile ...]

positional arguments::

  fitsfile [fitsfile ...]

The [options] are described in the following sections.

Informational switches
++++++++++++++++++++++
**-h, --help**
    show the help message and exit

**-v, --version**
    show program's version number and exit

**-d, --displayflags**
    Display all parsed option flags and exit.

    When specified, this switch will present the user with a table of all 
    parsed arguments and then exit without running. This allows the user to 
    check that the configuration is as intended. The table provides a convenient
    view of all passed and default values. Unless a user has specified a 
    recipe (-r, --recipe), 'recipename' indicates 'None' because at this point, 
    the Recipe System has not yet been engaged and a default recipe not yet
    determined.

    Eg.,::

       $ reduce -d --logmode console fitsfile.fits
       
       --------------------   switches, vars, vals  --------------------
       
       Literals			var 'dest'		Value
       -----------------------------------------------------------------
       ['--invoked'] 	        :: invoked 		:: False
       ['--addprimset'] 	:: primsetname 		:: None
       ['-d', '--displayflags'] :: displayflags 	:: True
       ['-p', '--param'] 	:: userparam 		:: None
       ['--logmode'] 		:: logmode 		:: ['console']
       ['-r', '--recipe'] 	:: recipename 		:: None
       ['--throw_descriptor_exceptions'] :: throwDescriptorExceptions :: False
       ['--logfile'] 		:: logfile 		:: reduce.log
       ['-t', '--astrotype'] 	:: astrotype 		:: None
       ['--override_cal'] 	:: user_cals 		:: None
       ['--context'] 		:: running_contexts 	:: None
       ['--calmgr'] 		:: cal_mgr 		:: None
       ['--suffix'] 		:: suffix 		:: None
       ['--loglevel'] 		:: loglevel 		:: stdinfo
       -----------------------------------------------------------------
       
       Input fits file(s):	fitsfile.fits

Configuration Switches, Options
+++++++++++++++++++++++++++++++
**--addprimset <PRIMSETNAME>** 
    Add this path to user-supplied primitives for reduction. eg., path to a 
    primitives module.

**--calmgr <CAL_MGR>**
    This is a URL specifying a calibration manager service. A calibration manager 
    overides Recipe System table. Not available outside Gemini operations.

**--context <RUNNING_CONTEXTS>**
    Use <RUNNING_CONTEXTS> for primitives sensitive to context. Eg., --context QA
    When not specified, the context defaults to 'QA'. 

**--invoked**
    Boolean indicating that reduce was invoked by the control center.

**--logmode <LOGMODE>**
    Set logging mode. One of 'standard', 'console', 'quiet', 'debug', or 'null',
    where 'console' writes only to screen and 'quiet' writes only to the log
    file. Default is 'standard'.

**--logfile <LOGFILE>**
    Set the log file name. Default is 'reduce.log' in the current directory.

**--loglevel <LOGLEVEL>**
    Set the verbose level for console logging. One of
    'critical', 'error', 'warning', 'status', 'stdinfo', 'fullinfo', 'debug'.
    Default is 'stdinfo'.

**--override_cal <USER_CALS [USER_CALS ...]>**
    The option allows users to provide their own calibrations to ``reduce``.
    Add a calibration to User Calibration Service. 
    '--override_cal CALTYPE:CAL_PATH'
    Eg.,:

      --override_cal processed_arc:wcal/gsTest_arc.fits

**-p <USERPARAM [USERPARAM ...]>, --param <USERPARAM [USERPARAM ...]>**
    Set a primitive parameter from the command line. The form '-p par=val' sets 
    the parameter in the reduction context such that all primitives will 'see' it.
    The form ::

    -p ASTROTYPE:primitivename:par=val

    sets the parameter such that it applies only when the current reduction type 
    (type of current reference image) is 'ASTROTYPE' and the primitive is 
    'primitivename'. Separate parameter-value pairs by whitespace: 
    (eg. '-p par1=val1 par2=val2')

    See Sec. :ref:`userpars`, for more information on these values.

**-r <RECIPENAME>, --recipe <RECIPENAME>**
    Specify an explicit recipe to be used rather than internally determined by
    a dataset's <ASTROTYPE>. Default is None and later determined by the Recipe 
    System based on the AstroDataType.

**-t <ASTROTYPE>, --astrotype <ASTROTYPE>**
    Run a recipe based on this AstroDataType, which overrides default type or 
    begins without initial input. Eg., recipes that begin with primitives that 
    acquire data. ``reduce`` default is None and determined internally.

**--suffix <SUFFIX>**
    Add 'suffix' to output filenames at end of reduction.

**--throw_descriptor_exceptions**
    Boolean indicating descriptor exceptions are to be raised. This is a 
    development switch.

Nominal Usage
+++++++++++++
The minimal call for reduce can be ::

   $ reduce <dataset.fits>

While this minimal call is available at the Gemini Observatory, if a calibration 
service is unavailable to the user -- likely true for most users -- users should 
call ``reduce`` on a specified dataset by providing calibration files with the 
--overrride_cal option. For example::

  $ reduce --override_cal processed_arc:wcal/gsTest_arc.fits <dataset.fits>

Such a command for complex processing of data is possible because AstroData 
and the Recipe System do all the necessary work in determining how the data are to 
be processed, which is critcially based upon the determination of the `typeset` 
that applies to that data.

Without any user-specified recipe (-r --recipe), the default recipe is 
``qaReduce``, which is defined for various AstroDataTypes and currently used at 
the summit. For example, the ``qaReduce`` recipe for a GMOS_IMAGE specifies that 
the following primitives are called on the data::

 prepare
 addDQ
 addVAR
 detectSources
 measureIQ
 measureBG
 measureCCAndAstrometry
 overscanCorrect
 biasCorrect
 ADUToElectrons
 addVAR
 flatCorrect
 mosaicDetectors
 makeFringe
 fringeCorrect
 detectSources
 measureIQ
 measureBG
 measureCCAndAstrometry
 addToList

The point here is not to overwhelm readers with a stack of primitive names, but 
to present both the default pipeline processing that the above simple ``reduce`` 
command invokes and to demonstrate how much the ``reduce`` interface abstracts 
away the complexity of the processing that is engaged with the simplist of 
commands.

.. _userpars:

Overriding Primitive Parameters
+++++++++++++++++++++++++++++++

In some cases, users may wish to change the functional behaviour of certain 
processing steps, i.e. change default behaviour of primitive 
functions.

Each primitive has a set of pre-defined parameters, which are used to control 
functional behaviour of the primitive. Each defined parameter has a "user 
override" token, which indicates that a particular parameter may be overridden 
by the user. Users can adjust parameter values from the reduce command line with 
the option,

    **-p, --param**

If permitted by the "user override" token, parameters and values specified 
through the **-p, --param** option will `override` the defined 
parameter default value and may alter default behaviour of the primitive 
accessing this parameter. A user may pass several parameter-value pairs with this 
option.

Eg.::

  $ reduce -p par1=val1 par2=val2 [par3=val3 ... ] <fitsfile1.fits>

For example, some photometry primitives perform source detection on an image. The 
'detection threshold' has a defined default, but a user may alter this parameter
default to change the source detection behaviour::

  $ reduce -p threshold=4.5 <fitsfile.fits>

.. dev of parameter viewer ..

.. _atfile:

The @file facility
++++++++++++++++++

The reduce command line interface supports what might be called an 'at-file' 
facility (users and readers familiar with IRAF will recognize this facility). This
facility allows users to provide any and all command line options and flags 
to ``reduce`` via in a single acsii text file.

By passing an @file to ``reduce`` on the command line, users can encapsulate all 
the options and positional arguments they might wish to specify in a single 
@file. It is possible to use multiple @files and even to embed one or more 
@files in another. The parser opens all files sequentially and parses
all arguments in the same manner as if they were specified on the command line.
Essentially, an @file is some or all of the command line and parsed identically.

To illustrate the convenience provided by an '@file', let us begin with an 
example `reduce` command line that has a number of arguments::

  $ reduce -p GMOS_IMAGE:contextReport:tpar=100 GMOS_IMAGE:contextReport:report_inputs=True 
    -r recipe.ArgsTest --context qa S20130616S0019.fits N20100311S0090.fits

Ungainly, to be sure. Here, two (2) `user parameters` are being specified 
with **-p**, a `recipe` with **-r**, and a `context` argument is specified 
to be **qa** . This can be wrapped in a plain text @file called 
`reduce_args.par`::

   S20130616S0019.fits
   N20100311S0090.fits
   --param
   GMOS_IMAGE:contextReport:tpar=100
   GMOS_IMAGE:contextReport:report_inputs=True
   -r recipe.ArgsTests
   --context qa

This then turns the previous reduce command line into something a little more 
`keyboard friendly`::

  $ reduce @reduce_args.par

The order of these arguments is irrelevant. The parser will figure out what is 
what. The above file could be thus written like::

  -r recipe.ArgsTests
  --param
  GMOS_IMAGE:contextReport:tpar=100
  GMOS_IMAGE:contextReport:report_inputs=True
  --context qa
  S20130616S0019.fits
  N20100311S0090.fits

.. note:: Comments are accommodated, both line and in-line. '=' signs `may` be 
	  used but this has meaning only for arguments that expect unitary values. 
	  The '=' is really quite unnecessary.

	  White space is the only significant separator of arguments: spaces, 
	  tabs, newlines are all equivalent when argument parsing. This means 
	  the user can 'arrange' their @file for clarity.

	  Eg., a more readable version of the above file might be written as::

	    # reduce parameter file
	    # yyyy-mm-dd
	    # GDPSG 
	    
	    # Spec the recipe
	    -r 
	        recipe.ArgsTests  # test recipe
	    
	    # primitive parameters here
	    # These are 'untyped', i.e. global
	    --param
	        tpar=100
	        report_inputs=True
	    
	    --context 
	        qa                # QA context
	    
	    S20130616S0019.fits
	    N20100311S0090.fits

All the above  examples of ``reduce_args.par`` are equivalently parsed. Which, 
of course, users may check by adding the **-d** flag::

  $ reduce -d @redpars.par
  
  --------------------   switches, vars, vals  --------------------

  Literals			var 'dest'		Value
  -----------------------------------------------------------------
  ['--invoked'] 		:: invoked 		:: False
  ['--addprimset'] 		:: primsetname 		:: None
  ['-d', '--displayflags'] 	:: displayflags 	:: True
  ['-p', '--param'] 		:: userparam 		:: ['tpar=100', 'report_inputs=True']
  ['--logmode'] 		:: logmode 		:: standard
  ['-r', '--recipe'] 		:: recipename 		:: ['recipe.ArgTests']
  ['--throw_descriptor_exceptions'] :: throwDescriptorExceptions 	:: False
  ['--logfile'] 		:: logfile 		:: reduce.log
  ['-t', '--astrotype'] 	:: astrotype 		:: None
  ['--override_cal'] 		:: user_cals 		:: None
  ['--context'] 		:: running_contexts 	:: ['QA']
  ['--calmgr'] 			:: cal_mgr 		:: None
  ['--suffix'] 			:: suffix 		:: None
  ['--loglevel'] 		:: loglevel 		:: stdinfo
  -----------------------------------------------------------------

  Input fits file(s):	S20130616S0019.fits
  Input fits file(s):	N20100311S0090.fits

Recursive @file processing
++++++++++++++++++++++++++

As implemented, the @file facility will recursively handle, and process 
correctly, other @file specifications that appear in a passed @file or 
on the command line. For example, we may have another file containing a 
list of fits files, separating the command line flags from the positional 
arguments.

We have a plain text 'fitsfiles' containing the line::

  test_data/S20130616S0019.fits

We can indicate that this file is to be consumed with the prefix character 
"@" as well. In this case, the 'reduce_args.par' file could thus appear::

  # reduce test parameter file 
  
  @fitsfiles       # file with fits files
  
  # AstroDataType
  -t GMOS_IMAGE
  
  # primitive parameters.  
  --param
      report_inputs=True
      tpar=99
      FOO=BAR

  # Spec the recipe
  -r recipe.ArgTests

The parser will open and read the @fitsfiles, consuming those lines in the 
same way as any other command line arguments. Indeed, such a file need not only 
contain fits files (positional arguments), but other arguments as well. This is 
recursive. That is, the @fitsfiles can contain other at-files", which can contain 
other "at-files", which can contain ..., `ad infinitum`. These will be processed 
serially.

As stipulated earlier, because the @file facility provides arguments equivalent 
to those that appear on the command line, employment of this facility means that 
a reduce command line could assume the form::

   $ reduce @parfile @fitsfiles

or equally::

   $ reduce @fitsfiles @parfile

where 'parfile' could contain the flags and user parameters, and 'fitsfiles' 
could contain a list of datasets.

Eg., fitsfiles comprises the one line::

  test_data/N20100311S0090.fits

while parfile holds all other specifications::

  # reduce test parameter file
  # GDPSG
  
  # AstroDataType
  -t GMOS_IMAGE
  
  # primitive parameters.
  --param 
      report_inputs=True
      tpar=99            # This is a test parameter
      FOO=BAR            # This is a test parameter
  
  # Spec the recipe
  -r recipe.ArgTests


Overriding @file values
+++++++++++++++++++++++
The ``reduce`` application employs a customized command line parser such that 
the command line option 

**-p** or **--param**

will accumulate a set of parameters `or` override a particular parameter. 
This may be seen when a parameter is specified in a user @file and then 
specified on the command line. For unitary value arguments, the command line 
value will `override` the @file value.

It is further specified that if one or more datasets (i.e. positional arguments) 
are passed on the command line, `all fits files appearing as positional arguments` 
`in the parameter file will be replaced by the command line arguments.`

Using the parfile above,

Eg. 1)  Accumulate a new parameter::

  $ reduce @parfile --param FOO=BARSOOM
  
  parsed options:
  --------------------
  AstroDataType: GMOS_IMAGE
  FITS files:    ['S20130616S0019.fits', 'N20100311S0090.fits']
  Parameters:    tpar=100, report_inputs=True, FOO=BARSOOM
  RECIPE:        recipe.ArgsTest

Eg. 2) Override a parameter in the @file::

  $ reduce @parfile --param tpar=99
  
  parsed options:
  --------------------
  AstroDataType: GMOS_IMAGE
  FITS files:    ['S20130616S0019.fits', 'N20100311S0090.fits']
  Parameters:    tpar=99, report_inputs=True
  RECIPE:        recipe.ArgsTest

Eg. 3) Override the recipe::

  $ reduce @parfile -r=recipe.FOO
  
  parsed options:
  --------------------
  AstroDataType:    GMOS_IMAGE
  FITS files:       ['S20130616S0019.fits', 'N20100311S0090.fits']
  Parameters:       tpar=100, report_inputs=True
  RECIPE:           recipe.FOO

Eg. 4) Override a recipe and specify another fits file ::

  $ reduce @parfile -r=recipe.FOO test_data/N20100311S0090_1.fits
  
  parsed options:
  --------------------
  AstroDataType:    GMOS_IMAGE
  FITS files:       ['test_data/N20100311S0090_1.fits']
  Parameters:       tpar=100, report_inputs=True
  RECIPE:           recipe.FOO


Application Programming Interface (API)
---------------------------------------
.. note:: The following sections discuss and describe programming interfaces
          available on ``reduce`` and the underlying class Reduce.

The ``reduce`` application is essentially a skeleton script providing the 
described command line interface. After parsing the command line, the script 
then passes the parsed arguments to its main() function, which in turn calls 
the Reduce() class constructor with "args". Class Reduce() is defined 
in the module ``coreReduce.py``. ``reduce`` and class Reduce are both 
scriptable, as the following discussion will illustrate.

.. _main:

reduce.main()
+++++++++++++

The main() function of reduce receives one (1) parameter that is a Namespace 
object as returned by a call on ArgumentParser.parse_args(). Specific to reduce, 
the caller can supply this object by a call on the parseUtils.buildParser() 
function, which returns a fully defined reduce parser. As usual, the parser 
object should then be called with the parse_args() method to return a valid 
reduce parser Namespace. Since there is no interaction with sys.argv, as in 
a command line call, all Namespace attributes have only their defined default 
values. It is for the caller to set these values as needed.

As the example below demonstrates, once the "args" Namespace object is 
instantiated, a caller can set any arguments as needed. Bu they must be set 
to the correct type. The caller should examine the various "args" types to 
determine how to set values. For example, args.files is type list, whereas 
args.recipename is type string.

Eg.,

    >>> from astrodata.adutils.reduceutils import reduce
    >>> from astrodata.adutils.reduceutils import parseUtils
    >>> args = parseUtils.buildParser("Reduce,v2.0").parse_args()
    >>> args.files
    []
    >>> args.files.append('S20130616S0019.fits')
    >>> args.recipename = "recipe.FOO"
    >>> reduce.main(args)
    --- reduce, v2.0 ---
    Starting Reduction on set #1 of 1
    Processing dataset(s):
    S20130616S0019.fits
    ...

Processing will proceed as usual.

Class Reduce and the runr() method
++++++++++++++++++++++++++++++++++

Class Reduce is defined in ``astrodata.adutils.reduceutils`` module, 
``coreReduce.py``.

The reduce.main() function serves mainly as a callable for the command line 
interface. While main() is callable by users supplying the correct "args" 
parameter (See :ref:`main`), the Reduce() class is also callable and 
can be used directly, and more appropriately. Callers need not supply an "args" 
parameter to the class constructor. The instance of Reduce will have all the 
same arguments as in a command line scenario, available as attributes on the 
instance. Once an instance of Reduce() is instantiated and instance attributes 
set as needed, there is one (1) method to call, **runr()**. This is the only 
public method on the class.

.. note:: When using Reduce() directly, callers must configure their own logger. 
	  Reduce() does not configure logutils prior to using a logger as 
	  returned by logutils.get_logger(). The following example will illustrate 
	  how this is easily done. It is `highly recommended` that callers
	  configure the logger. 

Eg.,

>>> from astrodata.adutils.reduceutils.coreReduce import Reduce
>>> reduce = Reduce()
>>> reduce.files
[]
>>> reduce.files.append('S20130616S0019.fits')
>>> reduce.files
['S20130616S0019.fits']

Once an instance of Reduce has been made, callers can then configure logutils 
with the appropriate settings supplied on the instance. This is precisely what 
``reduce`` does when it configures logutils.

>>> from astrodata.adutils import logutils
>>> logutils.config(file_name=reduce.logfile, mode=reduce.logmode, 
                    console_lvl=reduce.loglevel)

At this point, the caller is able to call the runr() method on the "reduce" 
instance.

   >>> reduce.runr()
   All submitted files appear valid
   Starting Reduction on set #1 of 1
   Processing dataset(s):
   S20130616S0019.fits
   ...

Processing will then proceed in the usual manner.
