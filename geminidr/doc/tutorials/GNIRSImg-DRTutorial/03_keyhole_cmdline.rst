.. keyhole_cmdline.rst

.. include:: DRAGONSlinks.txt

.. _keyhole_cmdline:

**********************************************************************
Example 1-A: Point source through keyhole - Using the "reduce" command
**********************************************************************

In this example we will reduce a GNIRS keyhole imaging observation of a point
source using the "|reduce|" command that is operated directly from the unix
shell.  Just open a terminal to get started.

This observation is a simple dither on target.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to the :ref:`datasetup` for the links and simple instructions.

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

Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and a local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows "|reduce|" to make requests for matching **processed**
calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, create or edit the configuration file ``rsys.cfg`` as
follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/gnirsimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

.. note:: ``~`` in the path above refers to your home directory.  Also, don't
    miss the dot in ``.geminidr``.

Then initialize the calibration database::

    caldb init

That's it.  It is ready to use.

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file from the database (it will **not**
remove the file on disk.)  (See the "|caldb|" documentation for more details.)


Create file lists
=================
.. |astrouser_link| raw:: html

   <a href="https://astrodata-user-manual.readthedocs.io/" target="_blank">Astrodata User Manual</a>

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrouser_link| for information about Astrodata.)

First, navigate to the ``playground`` directory in the unpacked data package.

A list of the darks
-------------------
There is only one set of 60-second darks in the data package.  To create the
list, one simply need to select on the ``DARK`` tag::

    dataselect ../playdata/*.fits --tags DARK -o darks60.lis

If there was a need to select specifically on the 60-second darks, the
command would use the ``exposure_time`` descriptor::

    dataselect ../playdata/*.fits --tags DARK --expr='exposure_time==60' -o darks60.lis

A list for the flats
--------------------
The flats are a sequence of lamp-on and lamp-off exposures.  We just send all
of them to one list.

::

    dataselect ../playdata/*.fits --tags FLAT -o flats.lis

A list for the science observations
-----------------------------------
The science frames are all the ``IMAGE`` non-``FLAT`` frames in the data
package.  They are also the ``J`` filter images that are non-``FLAT``. And
they are the ones with an object name ``GRB120116A``.  Those are all valid
ways to select the science observations.  Here we show all three ways as
examples; of course, just one is required.

::

    dataselect ../playdata/*.fits --tags IMAGE --xtags FLAT -o target.lis

    dataselect ../playdata/*.fits --xtags FLAT --expr='filter_name=="J"' -o target.lis

    dataselect ../playdata/*.fits --expr='object=="GRB120116A"' -o target.lis

Pick the one you prefer, they all yield the same list.



Master Dark
===========
We first create the master dark for the science target, then add it to the
calibration database.  The name of the output master dark,
``N20120102S0538_dark.fits``, is written to the screen at the end of the
process.

::

    reduce @darks60.lis
    caldb add N20120102S0538_dark.fits

The ``@`` character before the name of the input file is the "at-file" syntax.
More details can be found in the |atfile| documentation.

.. note:: The file name of the output processed dark is the file name of the
          first file in the list with `_dark` appended as a suffix.  This the
          general naming scheme used by "|reduce|".


Master Flat Field
=================
A GNIRS master flat is created from a series of lamp-on and lamp-off exposures.
Each flavor is stacked, then the lamp-off stack is subtracted from the lamp-on
stack.

We create the master flat field and add it to the calibration database as
follow::

    reduce @flats.lis
    caldb add N20120117S0034_flat.fits


Science Observations
====================
The science target is a point source.  The sequence dithers on-target, moving
the source across the thin keyhole aperture.  The sky frames for each
science image will be the adjacent dithered frames obtained within a certain
time limit.  The default for GNIRS keyhole images is "within 600 seconds".
This can be seen by using "|showpars|"::

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

    reduce @target.lis --user_cal processed_dark:N20120102S0538_dark.fits

The output stack units are in electrons (header keyword BUNIT=electrons).
The output stack is stored in a multi-extension FITS (MEF) file.  The science
signal is in the "SCI" extension, the variance is in the "VAR" extension, and
the data quality plane (mask) is in the "DQ" extension.

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

