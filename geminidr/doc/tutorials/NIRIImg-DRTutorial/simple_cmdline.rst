.. simple_cmdline.rst

.. _simple_cmdline:

*****************************************
Simple example using the "reduce" command
*****************************************

The dataset
===========
This is a NIRI imaging observation of the an extended source, a galaxy showing
as a dense field of stars.  The observation sequence uses offset to sky to
monitor it.

The calibrations we use here include:

* Darks for the science and sky offset frames.
* Flats, as a sequence of lamps-on and lamps-off exposures.
* Short darks to use with the flats to create a bad pixel mask.
* A set of standard star observations.

Here are the files that need to be downloaded from the Gemini Observatory
Archive.

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

A note about finding the short darks.  Those are used solely to create a
fresh bad pixel mask (BPM).  In the archive, the calibration association
will not find those darks for you, you will need to search for them
explicitely. To do so,

* Set a date range around the dates of your science observations.
* Set **Instrument** to NIRI.
* Set the **Obs.Type** to DARK.
* Set the exposure time to 1 second.

All the data needed to run this tutorial are found in the tutorial's data
package (KL??? name of the package, with URL).  Download it and unpack it
somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvzf KL???

The datasets are found in the subdirectory ``niriimg_tutorial/playdata``, and we
will work in the subdirectory named ``niriimg_tutorial/playground``.


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
This is just a floating inaccuracy somewhere in the software that generates
the FITS file.  As far as we are concerned here in this tutorial, we are
dealing with 20-second and 1-second darks.  The tool ``dataselect`` is smart
enough to match those exposure times as "close enough".  So, in our selection
expression, we can use "1" and "20" and ignore the extra digits.

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
The standard sequence is a series of IMAGE that are not FLAT and identified
as "FS 17".  There are no keywords in the NIRI header identifying this target
as a special standard star target.  So we need to use the name to select only
that star and not our science target.

Flats are FLAT and IMAGE, this is why we need to exclude FLAT.

::

    dataselect ../playdata/*.fits --tags IMAGE --xtags FLAT --expr='object=="FS 17"' -o stdstar.lis



A list of the science sequence
------------------------------
The science frames are all the IMAGE non-FLAT that are also not the standard.
This translates to the following expression::

    dataselect ../playdata/*.fits --tags IMAGE --xtags FLAT --expr='object!="FS 17"' -o target.lis

One could use the name of the science target too.



Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows ``reduce`` to make requests for matching **processed**
calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/, edit the configuration file ``rsys.cfg`` as follow::

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


Bad Pixel Mask
--------------
The Gemini software comes with a static NIRI BPM that gets automatically added
to all the data as it gets processed.  The user can create from the flats and
short darks a *user* BPM that will be combined with the static BPM.  Using both
the static and a fresh BPM from recent data is a better representation of the
bad pixel.  It is a recommended step.

The flats must be passed first for ``reduce`` to select the recipe library
associated with NIRI flats.  We will not use the default recipe but rather
the special recipe from that library called ``makeProcessedBPM``.

The flats and the short darks are inputs.

::

    reduce @flats.lis @darks1s.lis -r makeProcessedBPM

The BPM produced is named ``N20160102S0373_bpm.fits``.

The local calibration manager does not yet support BPMs so we cannot added
it to the database.  It is a future feature.  We will have to pass it
manually to ``reduce`` to use it.


Master Flat Field
-----------------
A NIRI master flat is created from a series of lamp-on and lamp-off exposures.
Each flavor is stacked then the lamp-off stack is subtracted from the lamp-on
stack.

We create the master flat field and add it to the calibration manager as
follow::

    reduce @allflats.lis -p addDQ:user_bpm=N20160102S0373_bpm.fits
    caldb add N20160102S0373_flat.fits

Note how we pass in the BPM we created in the previous step.  The ``addDQ``
primitive, one of the primitives in the recipe, has an input parameter named
``user_bpm``.  We assign our BPM to that input parameter.


Standard Star
-------------
Reduce the standard star.  The flat field will be automatically picked
from the local calibration database.

::

    reduce @stdstar.lis -p addDQ:user_bpm=N20160102S0373_bpm.fits darkCorrect:do_dark=False


Science Target
--------------
Reduce the science target.  This is an extended source.  We need to turn off
the scaling of the sky because the target fills the field of view and does
not present a reasonable sky background.  If scaling is not turned off in
this particular case, it results in an oversubtraction of the sky frame.

The sky frame comes from off-target sky observations.  The software will
split the on-target and the off-target appropriately as long as the first
frame is on-target.

The master dark and master flats will be retrieved automatically from the
local calibration database.

::

    reduce @target.lis -p addDQ:user_bpm=N20160102S0373_bpm.fits skyCorrect:scale=False
