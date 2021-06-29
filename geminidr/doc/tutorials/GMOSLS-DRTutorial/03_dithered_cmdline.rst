.. 03_dithered_cmdline.rst

.. include:: DRAGONSlinks.txt

.. _dithered_cmdline:

*****************************************************************************
Example 1-A: Dithered Point Source Longslit - Using the "reduce" command line
*****************************************************************************

In this example we will reduce a GMOS longslit observation of a DB white
dwarf candidate
using the "|reduce|" command that is operated directly from the unix shell.
Just open a terminal and load the DRAGONS conda environment to get started.

This observation dithers along the slit and along the dispersion axis.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`datadithered`.

Here is a copy of the table for quick reference.

+---------------------+---------------------------------+
| Science             || S20171022S0087,89 (515 nm)     |
|                     || S20171022S0095,97 (530 nm)     |
+---------------------+---------------------------------+
| Science biases      || S20171021S0265-269             |
|                     || S20171023S0032-036             |
+---------------------+---------------------------------+
| Science flats       || S20171022S0088 (515 nm)        |
|                     || S20171022S0096 (530 nm)        |
+---------------------+---------------------------------+
| Science arcs        || S20171022S0092 (515 nm)        |
|                     || S20171022S0099 (530 nm)        |
+---------------------+---------------------------------+
| Standard (LTT2415)  || S20170826S0160 (515 nm)        |
+---------------------+---------------------------------+
| Standard biases     || S20170825S0347-351             |
|                     || S20170826S0224-228             |
+---------------------+---------------------------------+
| Standard flats      || S20170826S0161 (515 nm)        |
+---------------------+---------------------------------+
| Standard arc        || S20170826S0162 (515 nm)        |
+---------------------+---------------------------------+


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
    database_dir = <where_the_data_package_is>/gmosls_tutorial/playground

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

This data set contains science and calibration frames. For some programs, it
could contain different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  However, DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata.)

First, navigate to the ``playground/example1`` directory in the unpacked data package::

    cd <path>/gmosls_tutorial/playground/example1



Two lists for the biases
------------------------
The science observations and the spectrophotometric standard observations were
obtained using different
regions-of-interest (ROI).  So we will need two master biases, one "Full Frame"
for the science and one "Central Spectrum" for the standard.

We can use |dataselect| to select biases for each ROIs.

Given the data that we have in the ``playdata`` directory, we can create
our GMOS-S bias list using the tags and expression using
the ROI settings. Remember, this will always depend on what you have in your
raw data directory.  For easier selection criteria, you might want to
keep raw data from different programs in different directories.

First, let's see which biases we have for in our raw data
directory.

::

    dataselect ../../playdata/*.fits --tags BIAS | showd -d detector_roi_setting

    ---------------------------------------------------------
    filename                             detector_roi_setting
    ---------------------------------------------------------
    ../../playdata/S20170825S0347.fits       Central Spectrum
    ../../playdata/S20170825S0348.fits       Central Spectrum
    ../../playdata/S20170825S0349.fits       Central Spectrum
    ../../playdata/S20170825S0350.fits       Central Spectrum
    ../../playdata/S20170825S0351.fits       Central Spectrum
    ../../playdata/S20170826S0224.fits       Central Spectrum
    ../../playdata/S20170826S0225.fits       Central Spectrum
    ../../playdata/S20170826S0226.fits       Central Spectrum
    ../../playdata/S20170826S0227.fits       Central Spectrum
    ../../playdata/S20170826S0228.fits       Central Spectrum
    ../../playdata/S20171021S0265.fits             Full Frame
    ../../playdata/S20171021S0266.fits             Full Frame
    ../../playdata/S20171021S0267.fits             Full Frame
    ../../playdata/S20171021S0268.fits             Full Frame
    ../../playdata/S20171021S0269.fits             Full Frame
    ../../playdata/S20171023S0032.fits             Full Frame
    ../../playdata/S20171023S0033.fits             Full Frame
    ../../playdata/S20171023S0034.fits             Full Frame
    ../../playdata/S20171023S0035.fits             Full Frame
    ../../playdata/S20171023S0036.fits             Full Frame


We can see the two groups that differ on their ROI.

::

    dataselect ../../playdata/*.fits --tags BIAS --expr='detector_roi_setting=="Central Spectrum"' -o biasesstd.lis
    dataselect ../../playdata/*.fits --tags BIAS --expr='detector_roi_setting=="Full Frame"' -o biasessci.lis


A list for the flats
--------------------
The GMOS longslit flats are not normally stacked.   The default recipe does
not stack the flats.  This allows us to use only one list of the flats.  Each
will be reduced individually, never interacting with the others.

If you have multiple programs and you want to reduce only the flats for that
program, you might want to use the ``program_id`` descriptor

Or, like here, we have only one set of flats, so we will just gather
them all together.

::

    dataselect ../../playdata/*.fits --tags FLAT -o flats.lis


A list for the arcs
-------------------
The GMOS longslit arcs are not normally stacked.  The default recipe does
not stack the arcs.  This allows us to use only one list of arcs.  Each will be
reduce individually, never interacting with the others.

The arcs normally share the ``program_id`` with the science observations if
you find that you need more accurate sorting.  We do not need it here.

::

    dataselect ../../playdata/*.fits --tags ARC -o arcs.lis


A list for the spectrophotometric standard star
-----------------------------------------------
If a spectrophotometric standard is recognized as such by DRAGONS, it will
receive the Astrodata tag ``STANDARD``.  All spectrophotometric standards
normally used at Gemini are in the DRAGONS list of recognized standards.

::

    dataselect ../../playdata/*.fits --tags STANDARD -o std.lis


A list for the science observation
----------------------------------

The science observations are what is left, anything that is not a calibration
or assigned the tag ``CAL``.

If we had multiple targets, we would need to split them into separate list. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../../playdata/*.fits --xtags CAL | showd -d object

    -----------------------------------------------
    filename                                 object
    -----------------------------------------------
    ../../playdata/S20171022S0087.fits   J2145+0031
    ../../playdata/S20171022S0089.fits   J2145+0031
    ../../playdata/S20171022S0095.fits   J2145+0031
    ../../playdata/S20171022S0097.fits   J2145+0031

Here we only have one object from the same sequence.  We would not need any
expression, just exclusing calibrations would be sufficient.  But we demonstrate
here how one would specify the object name for a more surgical selection.

::

    dataselect ../../playdata/*.fits --xtags CAL --expr='object=="J2145+0031"' -o sci.lis


Master Bias
===========
We create the master biases with the "|reduce|" command, then add them
to the local calibration manager with the |caldb| command.

::

    reduce @biasesstd.lis
    reduce @biasessci.lis
    caldb add *_bias.fits

The master biases are ``S20170825S0347_bias.fits`` and ``S20171021S0265_bias.fits``;
this information is in both the terminal log and the log file.  The ``@`` character
before the name of the input file is the "at-file" syntax. More details can be found in
the |atfile| documentation.

.. note:: The file name of the output processed bias is the file name of the
    first file in the list with ``_bias`` appended as a suffix.  This the
    general naming scheme used by "|reduce|".


Master Flat Field
=================
GMOS longslit flat field are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.  The
matching flat nearest in time to the target observation is used to flat field
the target.  The central wavelength, filter, grating, binning, gain, and
read speed must match.

Because of the flexure, GMOS longslit flat field are not stacked.  Each is
reduced and used individually.  The default recipe takes that into account.

We can send all the flats, regardless of characteristics, to |reduce| and each
will be reduce individually.  When a calibration is needed, in this case, a
master bias, the best match will be obtained automatically from the local
calibration manager.

::

    reduce @flats.lis --ql
    caldb add *_flat.fits

We can bulk-add the master flats to the local calibration manager as shown
above.

.. note:: GMOS longslit reduction is currently available only for quicklook
   reduction.  The science quality recipes do not exist, hence the use of the
   ``--ql`` flag to activate the "quicklook" recipes.


Processed Arc - Wavelength Solution
===================================
GMOS longslit arc can be obtained at night with the observation sequence,
if requested by the program, but are often obtained at the end of the night
or the following afternoon instead.  Like the spectroscopic flats, they are not
stacked which means that they can be sent to reduce all to together and will
be reduced individually.

The wavelength solution is automatically calculated and has been found to be
quite reliable.  There might be cases where it fails; inspect the
``*_mosaic.pdf`` plot and the RMS of ``determineWavelengthSolution`` in the
logs to confirm a good solution.

::

    reduce @arcs.lis --ql
    caldb add *_arc.fits

.. note:: Failures of the wavelength solution calculation are not easy to fix
   in quicklook mode.  It might be better to simply not use the arc at all and
   rely on the approximate solution instead.  When the science quality package
   is released, there will be interactive tools to fix a bad solution.
   Remember, this version only offers quicklook reduction for GMOS longslit.


Processed Standard - Sensitivity Function
=========================================
The GMOS longslit spectrophotometric standards are normally taken when there
is a hole in the queue schedule, often when the weather is not good enough
for science observations.  One standard per configuration, per program is
the norm.  If you dither along the dispersion axis, most likely only one
of the positions will have been used for the spectrophotometric standard.
This is normal for baseline calibrations at Gemini.  The standard is used
to calculate the sensitivity function.  It has been shown that a difference of
10 or so nanometers in central wavelength setting does not significantly impact
the spectrophotometric calibration.

The reduction of the standard will be using a master bias, a master flat,
and a processed arc.  If those have been added to the local calibration
manager, they will be picked up automatically.

::

    reduce @std.lis --ql
    caldb add *_standard.fits

We currently do not have tools to inspect the spectra or the calculated
sensitivity function.  In the Tips and Tricks chapter, we show a way to plot
them using matplotlib: :ref:`plot_1d`.

KL?????   Tips and tricks to plot sensfunc


Science Observations
====================
The science target is a DB white dwarf candidate.  The sequence has four images
that were dithered spatially and along the dispersion axis.  DRAGONS will
register the four images in both directions, align and stack them before
extracting the 1-D spectrum.

.. note::  In this observation, there is only one source to extract.  If there
   were multiple sources in slits, regardless of whether they are of interest to
   the program, DRAGONS will locate them, trace them, and extract them automatically.
   Each extracted spectrum is stored in an individual extension in the output
   multi-extension FITS file.

This is what one raw image looks like.

_graphics/rawscience.png

With the master bias, the master flat, the processed arcs (one for each of the
grating position, aka central wavelength), and the processed standard in the
local calibration manager, to reduce the science observations and extract the 1-D
spectrum, one only needs to do as follow.

::

    reduce @sci.lis --ql

This produces a 2-D spectrum (``N20180526S1024_2D.fits``) which has been
biased, flat fielded, QE-corrected, wavelength-calibrated, corrected for
distortion, sky subtracted, and stacked.  It also produces the 1-D spectra
extracted from that 2-D spectrum (``N20180526S1024_1D.fits``).  Each 1-D
spectrum flux calibrated with the sensitivity function from the
spectrophotometric standard. The 1-D spectra are stored as 1-D FITS images in
extensions of the output Multi-Extension FITS file.

This is what the 2-D spectrum looks like.

::

    reduce -r display N20180526S1024_2D.fits

_graphics/???

The aperture found are list in the log for the ``findApertures`` just before
the call to ``traceApertures``.  Information about the apertures are also
available in the header of each extracted spectrum.
And this is what the 1-D flux-calibrated spectrum of the primary target looks
like.

_graphics/???

To learn how to plot a 1-D spectrum with matplotlib, see Tips and Tricks ???KL???.
