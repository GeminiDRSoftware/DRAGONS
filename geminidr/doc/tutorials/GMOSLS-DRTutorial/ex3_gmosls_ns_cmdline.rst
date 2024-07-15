.. ex3_gmosls_ns_cmdline.rst

.. _ns_cmdline:

**************************************************************************
Example 3 - Nod-and-Shuffle Point Source - Using the "reduce" command line
**************************************************************************


In this example we will reduce a GMOS longslit nod-and-shuffle observation of
a quasar using the "|reduce|" command that is operated directly from
the unix shell. Just open a terminal and load the DRAGONS conda environment
to get started.

This observation dithers along the dispersion axis.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`ns_dataset`.

Here is a copy of the table for quick reference.

+---------------------+---------------------------------------------+
| Science             || N20190926S0130-32 (700 nm)                 |
|                     || N20190926S0137-39 (710 nm)                 |
+---------------------+---------------------------------------------+
| Science biases      || N20190926S0230-234                         |
+---------------------+---------------------------------------------+
| Science flats       || N20190926S0129,133 (700 nm)                |
|                     || N20190926S0136,140 (710 nm)                |
+---------------------+---------------------------------------------+
| Science arcs        || N20190926S0134 (700 nm)                    |
|                     || N20190926S0135 (710 nm)                    |
+---------------------+---------------------------------------------+
| Standard (G191B2B)  || N20190902S0046 (700 nm)                    |
+---------------------+---------------------------------------------+
| Standard biases     || N20190902S0089-093                         |
+---------------------+---------------------------------------------+
| Standard flats      || N20190902S0047 (700 nm)                    |
+---------------------+---------------------------------------------+
| Standard arc        || N20190902S0062 (700 nm)                    |
+---------------------+---------------------------------------------+
| BPM                 || bpm_20170306_gmos-n_Ham_12_full_12amp.fits |
+---------------------+---------------------------------------------+

Configuring the interactive interface
=====================================
In ``~/.dragons/``, add the following to the configuration file ``dragonsrc``::

    [interactive]
    browser = your_preferred_browser

The ``[interactive]`` section defines your preferred browser.  DRAGONS will open
the interactive tools using that browser.  The allowed strings are "safari",
"chrome", and "firefox".

Set up the Local Calibration Manager
====================================

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_cmdline`.


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

First, navigate to the ``playground`` directory in the unpacked data package::

    cd <path>/gmosls_tutorial/playground



Two lists for the biases
------------------------
The science observations and the spectrophotometric standard observations were
obtained using different regions-of-interest (ROI).  So we will need two master
biases, one "Full Frame" for the science and one "Central Spectrum" for the
standard.

We can use |dataselect| to select biases for each ROIs.

Given the data that we have in the ``playdata`` directory, we can create
our GMOS-N bias list using the tags and an expression that uses the ROI
settings. Remember, this will always depend on what you have in your raw data
directory.  For easier selection criteria, you might want to keep raw data
from different programs in different directories.

Let's see which biases we have for in our raw data directory.

::

    dataselect ../playdata/example3/*.fits --tags BIAS | showd -d detector_roi_setting

    ---------------------------------------------------------------
    filename                                   detector_roi_setting
    ---------------------------------------------------------------
    ../playdata/example3/N20190902S0089.fits       Central Spectrum
    ../playdata/example3/N20190902S0090.fits       Central Spectrum
    ../playdata/example3/N20190902S0091.fits       Central Spectrum
    ../playdata/example3/N20190902S0092.fits       Central Spectrum
    ../playdata/example3/N20190902S0093.fits       Central Spectrum
    ../playdata/example3/N20190926S0230.fits             Full Frame
    ../playdata/example3/N20190926S0231.fits             Full Frame
    ../playdata/example3/N20190926S0232.fits             Full Frame
    ../playdata/example3/N20190926S0233.fits             Full Frame
    ../playdata/example3/N20190926S0234.fits             Full Frame


We can see the two groups that differ on their ROI.  We can use that as a
search criterion for creating the list with |dataselect|

::

    dataselect ../playdata/example3/*.fits --tags BIAS --expr='detector_roi_setting=="Central Spectrum"' -o biasesstd.lis
    dataselect ../playdata/example3/*.fits --tags BIAS --expr='detector_roi_setting=="Full Frame"' -o biasessci.lis


A list for the flats
--------------------
The GMOS longslit flats are not normally stacked.   The default recipe does
not stack the flats.  This allows us to use only one list of the flats.  Each
will be reduced individually, never interacting with the others.

The flats used for nod-and-shuffle are normal flats.  The DRAGONS recipe will
"double" the flat and apply it to each beam.

::

    dataselect ../playdata/example3/*.fits --tags FLAT -o flats.lis


A list for the arcs
-------------------
The GMOS longslit arcs are not normally stacked.  The default recipe does
not stack the arcs.  This allows us to use only one list of arcs.  Each will be
reduced individually, never interacting with the others.

The arcs normally share the ``program_id`` with the science observations, if
you find that you need more accurate sorting.  We do not need it here.

::

    dataselect ../playdata/example3/*.fits --tags ARC -o arcs.lis


A list for the spectrophotometric standard star
-----------------------------------------------
If a spectrophotometric standard is recognized as such by DRAGONS, it will
receive the Astrodata tag ``STANDARD``.  All spectrophotometric standards
normally used at Gemini are in the DRAGONS list of recognized standards.

::

    dataselect ../playdata/example3/*.fits --tags STANDARD -o std.lis


A list for the science observations
-----------------------------------

The science observations are what is left, that is anything that is not a
calibration. Calibrations are assigned the astrodata tag ``CAL``, therefore
we can select against that tag to get the science observations.

If we had multiple targets, we would need to split them into separate list. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../playdata/example3/*.fits --xtags CAL | showd -d object

    --------------------------------------------------
    filename                                    object
    --------------------------------------------------
    ../playdata/example3/N20190926S0130.fits   J013943
    ../playdata/example3/N20190926S0131.fits   J013943
    ../playdata/example3/N20190926S0132.fits   J013943
    ../playdata/example3/N20190926S0137.fits   J013943
    ../playdata/example3/N20190926S0138.fits   J013943
    ../playdata/example3/N20190926S0139.fits   J013943


Here we only have one object from the same sequence.  We would not need any
expression, just excluding calibrations is sufficient.

::

    dataselect ../playdata/example3/*.fits --xtags CAL -o sci.lis


Bad Pixel Mask
==============
Starting with DRAGONS v3.1, the bad pixel masks (BPMs) are now handled as
calibrations.  They are downloadable from the archive instead of being
packaged with the software. They are automatically associated like any other
calibrations.  This means that the user now must download the BPMs along with
the other calibrations and add the BPMs to the local calibration manager.

See :ref:`getBPM` in :ref:`tips_and_tricks` to learn about the various ways
to get the BPMs from the archive.

To add the static BPM included in the data package to the local calibration
database:

::

    caldb add ../playdata/example3/bpm*.fits


Master Bias
===========
We create the master biases with the "|reduce|" command.  Because the database
was given the "store" option in the ``dragonsrc`` file, the processed biases
will be automatically added
to the database at the end of the recipe.

::

    reduce @biasesstd.lis
    reduce @biasessci.lis

The master biases are ``N20190902S0089_bias.fits`` and
``N20190926S0230_bias.fits``; this information is in both the terminal log
and the log file.  The ``@`` character before the name of the input file is
the "at-file" syntax. More details can be found in the |atfile| documentation.

.. note:: The file name of the output processed bias is the file name of the
    first file in the list with ``_bias`` appended as a suffix.  This the
    general naming scheme used by "|reduce|".

.. note:: If you wish to inspect the processed calibrations before adding them
    to the calibration database, remove the "store" option attached to the
    database in the ``dragonsrc`` configuration file.  You will then have to
    add the calibrations manually following your inspection, eg.

    ``caldb add *_bias.fits``


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

    reduce @flats.lis


Processed Arc - Wavelength Solution
===================================
GMOS longslit arc can be obtained at night with the observation sequence,
if requested by the program, but are often obtained at the end of the night
or the following afternoon instead. In this example, the arcs have been
obtained at night, as part of the sequence. Like the spectroscopic flats,
they are not stacked which means that they can be sent to reduce all together
and will be reduced individually.

The wavelength solution is automatically calculated and the algorithm has
been found to be quite reliable.  There might be cases where it fails;
inspect the ``*_wavelengthSolutionDetermined.pdf`` plot and the RMS of
``determineWavelengthSolution`` in the logs to confirm a good solution.

::

    reduce @arcs.lis



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

The reduction of the standard will be using a BPM, a master bias, a master flat,
and a processed arc.  If those have been added to the local calibration
manager, they will be picked up automatically.  The output of the reduction
includes the sensitivity function and will be added to the calibration
database automatically if the "store" option is set in the ``dragonsrc``
configuration file.

::

    reduce @std.lis

.. note:: If you wish to inspect the spectrum::

    dgsplot N20190902S0046_standard.fits 1

   where ``1`` is the aperture #1, the brightest target.
   To learn how to plot a 1-D spectrum with matplotlib using the WCS from a
   Python script, see Tips and Tricks :ref:`plot_1d`.

   The sensitivity function is stored within the processed standard spectrum.  To
   learn how to plot it, see Tips and Tricks :ref:`plot_sensfunc`.


Science Observations
====================
The science target is a quasar.  The sequence has six images in two groups
that were dithered along the dispersion axis.  DRAGONS will
remove the sky from the six images using the nod-and-shuffle beams.  The six
images will be register and stacked before extraction.

This is what one raw image looks like.

.. image:: _graphics/rawscience_ns.png
   :width: 600
   :alt: raw science image

With the master bias, the master flat, the processed arcs (one for each of the
grating position, aka central wavelength), and the processed standard in the
local calibration manager, one only needs to do as follows to reduce the
science observations and extract the 1-D spectrum.

::

    reduce @sci.lis

This produces a 2-D spectrum (``N20190926S0130_2D.fits``) which has been
bias corrected, flat fielded, QE-corrected, wavelength-calibrated, corrected for
distortion, sky-subtracted, the beams combined, and then all frames stacked.
It also produces the 1-D spectrum (``N20190926S0130_1D.fits``) extracted
from that 2-D spectrum.  The 1-D spectrum is flux calibrated with the
sensitivity function from the spectrophotometric standard. The 1-D spectra
are stored as 1-D FITS images in extensions of the output Multi-Extension
FITS file.

This is what the 2-D spectrum looks like.  Only the middle section is valid.

::

    reduce -r display N20190926S0130_2D.fits

.. note::

    ``ds9`` must be launched by the user ahead of running the display primitive.
    (``ds9&`` on the terminal prompt.)

.. image:: _graphics/2Dspectrum_ns.png
   :width: 600
   :alt: 2D stacked nod-and-shuffle spectrum

The apertures found are listed in the log for the ``findApertures`` primitive,
just before the call to ``traceApertures``.  Information about the apertures
are also available in the header of each extracted spectrum: ``XTRACTED``,
``XTRACTLO``, ``XTRACTHI``, for aperture center, lower limit, and upper limit,
respectively.

This is what the 1-D flux-calibrated spectrum of our sole target looks like.

::

    dgsplot N20190926S0130_1D.fits 1

.. image:: _graphics/1Dspectrum_ns.png
   :width: 600
   :alt: 1D spectrum

If you need an ascii representation of the spectum, you can use the primitive
``write1DSpectra`` to extract the values from the FITS file.

::

    reduce -r write1DSpectra N20190926S0130_1D.fits

The primitive outputs in the various formats offered by ``astropy.Table``.  To
see the list, use |showpars|.

::

    showpars N20190926S0130_1D.fits write1DSpectra

To use a different format, set the ``format`` parameters.

::

    reduce -r write1DSpectra -p format=ascii.ecsv extension='ecsv' N20190926S0130_1D.fits
