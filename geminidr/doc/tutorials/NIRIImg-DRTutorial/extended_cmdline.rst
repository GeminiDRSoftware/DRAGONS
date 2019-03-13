.. extended_cmdline.rst

.. _extended_cmdline:

*********************************************************
Example 1-A: Extended source - Using the "reduce" command
*********************************************************

In this example we will reduce a NIRI observation of an extended source using
the ``reduce`` command that is operated directly from the unix shell.  Just
open a terminal to get started.

This observation is a simple dither on target, a galaxy, with offset to sky.

The dataset
===========
If you have not already, download and unpackage the tutorial's data package.
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



Create file lists
=================
The first step is to create input file lists.  The tool ``dataselect`` helps
with that.  It uses Astrodata tags and descriptors to select the files and
send the information to a text file that can then be fed to ``reduce``.

First, navigate to the ``playground`` directory in the unpacked data package.

Two lists for the darks
-----------------------
We have two sets of darks; one set for the science frames, the 20-second darks,
and another for the BPM, the 1-second darks.  We will create two lists.

If you did not know the exposure times for the darks, you could have use a
combination of ``dataselect`` to select all the darks and feed that list to
``showd`` to show descriptor values, in this case ``exposure_time``.

.. highlight:: text

::

    dataselect ../playdata/*.fits --tags DARK | showd -d exposure_time

    N20160102S0423.fits: 20.002
    N20160102S0424.fits: 20.002
    N20160102S0425.fits: 20.002
    N20160102S0426.fits: 20.002
    N20160102S0427.fits: 20.002
    N20160102S0428.fits: 20.002
    N20160102S0429.fits: 20.002
    N20160102S0430.fits: 20.002
    N20160102S0431.fits: 20.002
    N20160102S0432.fits: 20.002
    N20160103S0463.fits: 1.001
    N20160103S0464.fits: 1.001
    N20160103S0465.fits: 1.001
    N20160103S0466.fits: 1.001
    N20160103S0467.fits: 1.001
    N20160103S0468.fits: 1.001
    N20160103S0469.fits: 1.001
    N20160103S0470.fits: 1.001
    N20160103S0471.fits: 1.001
    N20160103S0472.fits: 1.001

As one can see above the exposure times all have a small fractional increment.
This is just a floating point inaccuracy somewhere in the software that
generates the FITS file.  As far as we are concerned here in this tutorial,
we are dealing with 20-second and 1-second darks.  The tool ``dataselect`` is
smart enough to match those exposure times as "close enough".  So, in our
selection expression, we can use "1" and "20" and ignore the extra digits.

.. note:: If a perfect match to 1.001 were required, adding the option ``--strict`` in ``dataselect`` would ensure an exact match.

Let's create our two lists then.

::

    dataselect ../playdata/*.fits --tags DARK --expr='exposure_time==1' -o darks1s.lis
    dataselect ../playdata/*.fits --tags DARK --expr='exposure_time==20' -o darks20s.lis


A list for the flats
--------------------
The flats are a sequence of lamp-on and lamp-off exposures.  We just send all
that to one list.

::

    dataselect ../playdata/*.fits --tags FLAT -o flats.lis


A list for the standard star
----------------------------
The standard sequence is a series of datasets identified as "FS 17".  There
are no keywords in the NIRI header identifying this target as a special
standard star target.  So we need to use the target name to select only that
star and not our science target.

::

    dataselect ../playdata/*.fits --expr='object=="FS 17"' -o stdstar.lis



A list for the science observations
-----------------------------------
The science frames are all the IMAGE non-FLAT that are also not the standard.
Since flats are FLAT and IMAGE, we need to exclude the FLAT tag.

This translates to the following expression::

    dataselect ../playdata/*.fits --tags IMAGE --xtags FLAT --expr='object!="FS 17"' -o target.lis

One could use the name of the science target too, like we did for the selecting
the standard star observations in the previous section.



Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows ``reduce`` to make requests for matching **processed**
calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/`, create or edit the configuration file ``rsys.cfg`` as
follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/niriimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibration we are going to
send to it.

Then initialize the calibration database::

    caldb init

That's it.  It is ready to use.

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file from the database (it will not
remove the file on disk.)


Reduce the data
===============
We have our input lists, we have identified and initialzed the calibration
database, we are ready to reduce the data.

Please make sure that you are in the ``playground`` directory.


Master Dark
-----------
We first create the master dark for the science target, then add it to the
calibration database.  The name of the output master dark,
``N20160102S0423_dark.fits`` is written to the screen at the end of the process.

::

    reduce @darks20s.lis
    caldb add N20160102S0423_dark.fits

.. note:: The file name of the output processed dark is the file name of the first file in the list with `_dark` appended as a suffix.  This the general naming scheme used by `reduce`.


Bad Pixel Mask
--------------
The DRAGONS Gemini data reduction package comes with a static NIRI bad pixel
mask (BPM) that gets automatically added to all the NIRI data as it gets
processed.  The user can also create a supplemental, fresher BPM from the
flats and short darks.  It is fed to ``reduce`` as a *user* BPM that will
be combined with the static BPM.  Using both the static and a fresh BPM
from recent data is a better representation of the bad pixels.  It is an
optional but recommended step.

The flats must be passed first for ``reduce`` to select the recipe library
associated with NIRI flats.  We will not use the default recipe but rather
the special recipe from that library called ``makeProcessedBPM``.

The flats and the short darks are inputs.

::

    reduce @flats.lis @darks1s.lis -r makeProcessedBPM

The BPM produced is named ``N20160102S0373_bpm.fits``.

The local calibration manager does not yet support BPMs so we cannot add
it to the database.  It is a future feature.  We will have to pass it
manually to ``reduce`` later to use it.


Master Flat Field
-----------------
A NIRI master flat is created from a series of lamp-on and lamp-off exposures.
Each flavor is stacked, then the lamp-off stack is subtracted from the lamp-on
stack.

We create the master flat field and add it to the calibration manager as
follow::

    reduce @flats.lis -p addDQ:user_bpm=N20160102S0373_bpm.fits
    caldb add N20160102S0373_flat.fits

Note how we pass in the BPM we created in the previous step.  The ``addDQ``
primitive, one of the primitives in the recipe, has an input parameter named
``user_bpm``.  We assign our BPM to that input parameter.

To see the list of available input parameters and their defaults, use the
tool ``showpars``.  It needs the name of a file on which the primitive will
be run because the defaults are adjusted to match the input data.

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

    reduce @stdstar.lis -p addDQ:user_bpm=N20160102S0373_bpm.fits darkCorrect:do_dark=False


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

The master dark and master flats will be retrieved automatically from the
local calibration database. Again, the user BPM needs to be specified on
the command line.

::

    reduce @target.lis -p addDQ:user_bpm=N20160102S0373_bpm.fits skyCorrect:scale=False

.. image:: _graphics/extended_before.png
   :scale: 55%
   :align: left

.. image:: _graphics/extended_after.png
   :scale: 55%
   :align: left

The attentive reader will note that the reduced image is slightly larger
than the individual raw image. This is because of the telescope was dithered
between each observation leading to a slightly larger final field of view
than that of each individual image.  The stacked product is *not* cropped to
the common area, rather the image size is adjusted to include the complete
area covered by the whole sequence.  Of course the areas covered by less than
the full stack of images will have a lower signal-to-noise.