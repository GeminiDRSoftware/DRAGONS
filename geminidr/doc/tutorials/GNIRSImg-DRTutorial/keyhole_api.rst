.. keyhole_api.rst

.. _keyhole_api:

********************************************************************
Example 1-B: Point source through keyhole - Using the "Reduce" class
********************************************************************

A reduction can be initiated from the command line as shown in
:ref:`keyhole_cmdline` and it can also be done programmatically as we will
show here.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this example we replicate the command line reduction from
Example 1-A, this time using the Python interface instead of the command line.
Of course what is shown here could be packaged in modules for greater
automation.


The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`datakeyhole`.

Here is a copy of the table for quick reference.

+---------------+--------------------------------------------+
| Science       || N20120117S0014-33 (J-band, on-target)     |
+---------------+--------------------------------------------+
| Science darks || N20120102S0538-547 (60 sec, like Science) |
+---------------+--------------------------------------------+
| Flats         || N20120117S0034-41 (lamps-on)              |
|               || N20120117S0042-49 (lamps-off)             |
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
    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> from recipe_system import cal_service
    >>> from gempy.utils import logutils
    >>> from gempy.adlibrary import dataselect

Importing ``print_function`` is for compatibility with the Python 2.7 print
statement.  If you are working with Python 3, it is not needed, but importing
it will not break anything.

Create file lists
=================
.. |astrouser_link| raw:: html

   <a href="https://astrodata-user-manual.readthedocs.io/" target="_blank">Astrodata User Manual</a>

The first step is to create input file lists.  The tool ``dataselect`` helps
with that.  It uses Astrodata tags and descriptors to select the files and
store the filenames to a Python list that can then be fed to the ``Reduce``
class.  (See the |astrouser_link| for information about Astrodata.)

A list for the darks
--------------------
There is only one set of 60-second darks in the data package.  To create the
list, one simply need to select on the ``DARK`` tag::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> darks60 = dataselect.select_data(all_files, ['DARK'])

If there was a need to select specifically on the 60-second darks, the
command would use the ``exposure_time`` descriptor::

    >>> expression = 'exposure_time==60'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks60 = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)


A list for the flats
--------------------
The flats are a sequence of lamp-on and lamp-off exposures.  We just send all
of them to one list.

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> flats = dataselect.select_data(all_files, ['FLAT'])

A list for the science observations
-----------------------------------
The science frames are all the ``IMAGE`` non-``FLAT`` frames in the data
package.  They are also the ``J`` filter images that are non-``FLAT``. And
they are the ones with an object name ``GRB120116A``.  Those are all valid
ways to select the science observations.  Here we show all three ways as
examples; of course, just one is required.

::

    >>> all_files = glob.glob('../playdata/*.fits')

    >>> has_tags = ['IMAGE']
    >>> has_not_tags = ['FLAT']
    >>> target = dataselect.select_data(all_files, has_tags, has_not_tags)

    >>> has_tags = []
    >>> has_not_tags = ['FLAT']
    >>> expression = 'filter_name=="J"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> target = dataselect.select_data(all_files, has_tags, has_not_tags,
    ...                                 expression=parsed_expr)

    >>> expression = 'object=="GRB120116A"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> target = dataselect.select_data(all_files, [], [], expression=parsed_expr)

Pick the one you prefer, they all yield the same list.

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
    database_dir = <where_the_data_package_is>/gnirsimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

.. note:: ``~`` in the path above refers to your home directory.  Also, don't miss the dot in ``.geminidr``.

The calibration database is initialized and the calibration service is
configured like this::

    >>> caldb = cal_service.CalibrationService()
    >>> caldb.config()

    >>> caldb.init()

    >>> cal_service.set_calservice()

The calibration service is now ready to use.


Reduce the data
===============
We have our input filename lists, we have identified and initialized the
calibration database, we are ready to reduce the data.

Please make sure that you are still in the ``playground`` directory.

Set up the logging
------------------
First we quickly set up the logging::

    >>> logutils.config(file_name='gnirs_tutorial.log')


Master Dark
-----------
We first create the master dark for the science target, then add it to the
calibration database.  The name of the output master dark is
``N20120102S0538_dark.fits``.  The output is written to disk and its name is
stored in the ``Reduce`` instance.  The calibration service expects the
name of a file on disk.

::

    >>> reduce_darks = Reduce()
    >>> reduce_darks.files.extend(darks60)
    >>> reduce_darks.runr()

    >>> caldb.add_cal(reduce_darks.output_filenames[0])

.. note:: The file name of the output processed dark is the file name of the first file in the list with `_dark` appended as a suffix.  This the general naming scheme used by the ``Recipe System``.

Master Flat Field
-----------------
A GNIRS master flat is created from a series of lamp-on and lamp-off exposures.
Each flavor is stacked, then the lamp-off stack is subtracted from the lamp-on
stack.

We create the master flat field and add it to the calibration database as
follow::

    >>> reduce_flats = Reduce()
    >>> reduce_flats.files.extend(flats)
    >>> reduce_flats.runr()

    >>> caldb.add_cal(reduce_flats.output_filenames[0])


Science Observations
--------------------
The science target is a point source.  The sequence dithers on-target, moving
the source across the thin keyhole aperture.  The sky frames for each
science image will be the adjacent dithered frames obtained within a certain
time limit.  The default for GNIRS keyhole images is "within 600 seconds".
This can be seen by using the ``showpars`` command-line tool::

    showpars ../playdata/N20120117S0014.fits associateSky

.. image:: _graphics/showpars_associateSky.png
   :scale: 100%
   :align: center

Both the master dark and the master flat are in our local calibration
database.  For any other Gemini facility instrument, they would both be
retrieved automatically by the calibration manager.  However, GNIRS not being
an imager, and the keyhole being normally used only for acquisition, it turns
out that there are no calibration association rules between GNIRS keyhole images
and darks.  This is a recently discovered limitation that we plan to fix in
a future release.  In the meantime, we are not stuck, we can simply specify
the dark on the command line.  The flat will be retrieved automatically.

::

    >>> from recipe_system.utils.reduce_utils import normalize_ucals
    >>> mycalibrations = ['processed_dark:N20120102S0538_dark.fits']

    >>> reduce_target = Reduce()
    >>> reduce_target.files.extend(target)
    >>> ucals_dict = normalize_ucals(reduce_target.files, mycalibrations)
    >>> reduce_target.ucals = ucals_dict
    >>> reduce_target.runr()

Below are a raw image (top) and the final stacked image (bottom).  The stack
keeps all the pixels and is never cropped to only the common area. Of course
the areas covered by less than the full stack of images will have a lower
signal-to-noise.

.. image:: _graphics/gnirs_keyhole_before.png
   :scale: 60%
   :align: center

.. image:: _graphics/gnirs_keyhole_after.png
   :scale: 60%
   :align: center

