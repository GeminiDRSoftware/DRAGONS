.. extended_api.rst

.. .. include:: test.rst

.. _extended_api:

*******************************************************
Example 1-B: Extended source - Using the "Reduce" class
*******************************************************

A reduction can be initiated from the command line as shown in
:ref:`extended_cmdline` and it can also be done programmatically as we will
show here.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this example we replicate the command line reduction from
Example 1-A, this time using the Python interface instead of the command line.
Of course what is shown here could be packaged in modules for greater
automation.


|test|

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`dataextended`.

Here is a copy of the table for quick reference.

+---------------+--------------------------------------------+
| Science       || N20160102S0270-274 (on-target)            |
|               || N20160102S0275-279 (on-sky)               |
+---------------+--------------------------------------------+
| Science darks || N20160102S0423-432 (20 sec, like Science) |
+---------------+--------------------------------------------+
| Flats         || N20160102S0373-382 (lamps-on)             |
|               || N20160102S0363-372 (lamps-off)            |
+---------------+--------------------------------------------+
| Short darks   || N20160103S0463-472                        |
+---------------+--------------------------------------------+
| Standard star || N20160102S0295-299                        |
+---------------+--------------------------------------------+



Setting up
==========
First, navigate to the ``playground`` directory in the unpacked data package.

Then, we start Python and import the necessary modules, classes, and functions.

::

    % cd <path>/playground
    % python

::

    >>> from __future__ import print_function

    >>> import glob

    # DRAGONS imports
    >>> import astrodata
    >>> import gemini_instruments
    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> from recipe_system.cal_service import CalibrationService
    >>> from recipe_system.cal_service import set_calservice
    >>> from gempy.utils import logutils
    >>> from gempy.adlibrary import dataselect

Importing ``print_function`` is for compatibility with the Python 2.7 print
statement.  If you are working with Python 3, it is not needed, but importing
it will not break anything.



Create file lists
=================
The first step is to create input file lists.  The tool ``dataselect`` helps
with that.  It uses Astrodata tags and descriptors to select the files and
store the filenames to a Python list that can then be fed to the ``Reduce``
class.

Two lists for the darks
-----------------------
We have two sets of darks; one set for the science frames, the 20-second darks,
and another for making the BPM, the 1-second darks.  We will create two lists.

If you did not know the exposure times for the darks, you could have use
``dataselect`` as follow to see the exposure times of all the darks in the
directory.  We use the tag ``DARK`` and the descriptor ``exposure_time``.

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> all_darks = dataselect.select_data(all_files, ['DARK'])
    >>> for dark in all_darks:
    ...     ad = astrodata.open(dark)
    ...     print(dark, '  ', ad.exposure_time())
    ...
    ../playdata/N20160102S0423.fits    20.002
    ../playdata/N20160102S0424.fits    20.002
    ../playdata/N20160102S0425.fits    20.002
    ../playdata/N20160102S0426.fits    20.002
    ../playdata/N20160102S0427.fits    20.002
    ../playdata/N20160102S0428.fits    20.002
    ../playdata/N20160102S0429.fits    20.002
    ../playdata/N20160102S0430.fits    20.002
    ../playdata/N20160102S0431.fits    20.002
    ../playdata/N20160102S0432.fits    20.002
    ../playdata/N20160103S0463.fits    1.001
    ../playdata/N20160103S0464.fits    1.001
    ../playdata/N20160103S0465.fits    1.001
    ../playdata/N20160103S0466.fits    1.001
    ../playdata/N20160103S0467.fits    1.001
    ../playdata/N20160103S0468.fits    1.001
    ../playdata/N20160103S0469.fits    1.001
    ../playdata/N20160103S0470.fits    1.001
    ../playdata/N20160103S0471.fits    1.001
    ../playdata/N20160103S0472.fits    1.001

As one can see above the exposure times all have a small fractional increment.
This is just a floating point inaccuracy somewhere in the software that
generates the raw NIRI FITS files.  As far as we are concerned here in this
tutorial, we are dealing with 20-second and 1-second darks.  The function
``dataselect`` is smart enough to match those exposure times as "close enough".
So, in our selection expression, we can use "1" and "20" and ignore the
extra digits.

.. note:: If a perfect match to 1.001 were required, simply set the argument ``strict`` to ``True`` in ``dataselect.expr_parser``, eg. ``dataselect.expr_parser(expression, strict=True)``.

Let us create our two lists now.  The filenames will be stored in the variables
``darks1s`` and ``darks20s``.

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> expression = 'exposure_time==1'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks1s = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)

    >>> expression = 'exposure_time==20'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks20s = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)


A list for the flats
--------------------
The flats are a sequence of lamp-on and lamp-off exposures.  We just send all
of them to one list.

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> flats = dataselect.select_data(all_files, ['FLAT'])


A list for the standard star
----------------------------
The standard star sequence is a series of datasets identified as "FS 17".
There are no keywords in the NIRI header identifying this target as a special
standard star target.  We need to use the target name to select only
observations from that star and not our science target.

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> expression = 'object=="FS 17"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> stdstar = dataselect.select_data(all_files, expression=parsed_expr)


A list for the science observations
-----------------------------------
The science frames are all ``IMAGE`` non-``FLAT`` that are also not the
standard.  Since flats are tagged ``FLAT`` and ``IMAGE``, we need to exclude
the ``FLAT`` tag.

This translate to the following sequence::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> expression = 'object!="FS 17"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> has_tags = ['IMAGE']
    >>> has_not_tags = ['FLAT']
    >>> target = dataselect.select_data(all_files, has_tags, has_not_tags,
    ...                                 expression=parsed_expr)

One could have used the name of the science target too, like we did for
selecting the standard star observation in the previous section.  The example
above shows how to *exclude* a tag if needed and was considered more
educational.

Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and a local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows the ``Reduce`` instance to make requests for matching
**processed** calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, edit the configuration file ``rsys.cfg`` as follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/niriimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibration we are going to
send to it.

.. note:: ``~`` in the path above refers to your home directory.  Also, don't miss the dot in ``.geminidr``.

The calibration database is initialized and the calibration service is
configured like this::

    >>> from recipe_system.cal_service import CalibrationService
    >>> caldb = CalibrationService()
    >>> caldb.config()

    >>> caldb.init()

    >>> from recipe_system.cal_service import set_calservice
    >>> set_calservice()

The calibration service is now ready to use.


Reduce the data
===============
We have our input filename lists, we have identified and initialized the
calibration database, we are ready to reduce the data.

Please make sure that you are still in the ``playground`` directory.

Set up the logging
------------------
First we quickly set up the logging::

    >>> logutils.config(file_name='niri_tutorial.log')

Master Dark
-----------
We first create the master dark for the science target, then add it to the
calibration databae.  The name of the output master dark is
``N20160102S0423_dark.fits``.  The output is written to disk and its name is
stored in the ``Reduce`` instance.  The calibration service expects the
name of a file on disk.

::

    >>> reduce_darks = Reduce()
    >>> reduce_darks.files.extend(darks20s)
    >>> reduce_darks.runr()

    >>> caldb.add_cal(reduce_darks.output_filenames[0])

.. note:: The file name of the output processed dark is the file name of the first file in the list with `_dark` appended as a suffix.  This the general naming scheme used by the ``Recipe System``.


Bad Pixel Mask
--------------
The DRAGONS Gemini data reduction package, ``geminidr``, comes with a static
NIRI bad pixel mask (BPM) that gets automatically added to all the NIRI data
as they get processed.  The user can also create a *supplemental*, fresher BPM
from the flats and recent short darks.  That new BPM is later fed to
the reduction process as a *user BPM* to be combined with the static BPM.
Using both the static and a fresh BPM from recent data lead to a better
representation of the bad pixels.  It is an optional but recommended step.

The flats and the short darks are the inputs.

The flats must be passed first to the input list to ensure that the recipe
library associated with NIRI flats is selected. We will not use the default
recipe but rather the special recipe from that library called
``makeProcessedBPM``.


::

    >>> reduce_bpm = Reduce()
    >>> reduce_bpm.files.extend(flats)
    >>> reduce_bpm.files.extend(darks1s)
    >>> reduce_bpm.recipename = 'makeProcessedBPM'
    >>> reduce_bpm.runr()

    >>> bpm = reduce_bpm.output_filenames[0]

The BPM produced is named ``N20160102S0373_bpm.fits``.

The local calibration manager does not yet support BPMs so we cannot add
it to the database.  It is a future feature.  Until then we have to pass it
manually to the ``Reduce`` instance to use it, as we will show below.


Master Flat Field
-----------------
A NIRI master flat is created from a series of lamp-on and lamp-off exposures.
Each flavor is stacked, then the lamp-off stack is subtracted from the lamp-on
stack.

We create the master flat field and add it to the calibration manager as
follow::

::

    >>> reduce_flats = Reduce()
    >>> reduce_flats.files.extend(flats)
    >>> reduce_flats.uparms = [('addDQ:user_bpm', bpm)]
    >>> reduce_flats.runr()

    >>> caldb.add_cal(reduce_flats.output_filenames[0])

Note how we pass in the BPM we created in the previous step.  The ``addDQ``
primitive, one of the primitives in the recipe, has an input parameter named
``user_bpm``.  We assign our BPM to that input parameter.

To see the list of available input parameters and their defaults, use the
command line tool ``showpars`` from a terminal.  It needs the name of a file
on which the primitive will be run because the defaults are adjusted to match
the input data.

::

    showpars ../playdata/N20160102S0363.fits addDQ

.. image:: _graphics/showpars_addDQ.png
   :scale: 100%
   :align: center


Standard Star
-------------
The standard star is reduced more or less the same way as the science
target (next section) except that darks frames are not obtained for standard
star observations.  Therefore the dark correction needs to be turned off.

The processed flat field that we added earlier to the local calibration
database will be fetched automatically.  The user BPM (optional, but
recommended) needs to be specified by the user.

::

    >>> reduce_std = Reduce()
    >>> reduce_std.files.extend(stdstar)
    >>> reduce_std.uparms = [('addDQ:user_bpm', bpm)]
    >>> reduce_std.uparms.append(('darkCorrect:do_dark', False))
    >>> reduce_std.runr()


Science Observations
--------------------
The science target is an extended source.  We need to turn off
the scaling of the sky because the target fills the field of view and does
not represent a reasonable sky background.  If scaling is not turned off in
this particular case, it results in an over-subtraction of the sky frame.

The sky frame comes from off-target sky observations.  We feed the pipeline
all the on-target and off-target frames.  The software will split the
on-target and the off-target appropriately as long as the first frame is
on-target.

The master dark and the master flat will be retrieved automatically from the
local calibration database. Again, the user BPM needs to be specified as the
``user_bpm`` argument to ``addDQ``.

::

    >>> reduce_target = Reduce()
    >>> reduce_target.files.extend(target)
    >>> reduce_target.uparms = [('addDQ:user_bpm', bpm)]
    >>> reduce_target.uparms.append(('skyCorrect:scale', False))
    >>> reduce_target.runr()

.. image:: _graphics/extended_before.png
   :scale: 60%
   :align: left

.. image:: _graphics/extended_after.png
   :scale: 60%
   :align: left

The attentive reader will note that the reduced image is slightly larger
than the individual raw image. This is because of the telescope was dithered
between each observation leading to a slightly larger final field of view
than that of each individual image.  The stacked product is *not* cropped to
the common area, rather the image size is adjusted to include the complete
area covered by the whole sequence.  Of course the areas covered by less than
the full stack of images will have a lower signal-to-noise.