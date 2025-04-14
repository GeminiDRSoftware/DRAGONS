.. ex1_gnirsls_Kband_cmdline.rst

.. include:: symbols.txt

.. _Kband_cmdline:

**************************************************************************
Example 1 - K-band Longslit Point Source - Using the "reduce" command line
**************************************************************************

In this example, we will reduce the GNIRS K-band longslit observation of
"2MASSI J0605019-234226", an ultra-cool brown dwarf, using the "|reduce|"
command that is operated directly from the unix shell. Just open a terminal
and load the DRAGONS conda environment to get started.

This observation dithers along in a ABBA pattern.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`data_Kband`

Here is a copy of the table for quick reference.

+---------------------+----------------------------------------------+
| Science             || N20180106S0158-165                          |
+---------------------+----------------------------------------------+
| Science flats       || N20180106S0166                              |
+---------------------+----------------------------------------------+
| Science arcs        || N20180106S0172                              |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20100716_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

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


A list for the flats
--------------------
The GNRIS flats will be stack together.  Therefore it is important to ensure
that the flats in the list are compatible with each other.  You can use
`dataselect` to narrow down the selection as required.  Here, we have only
the flats that were taken with the science and we do not need extra selection
criteria.

::

    dataselect ../playdata/example1/*.fits --tags FLAT -o flats.lis

A list for the arcs
-------------------
The GNIRS longslit arc was obtained at the end of the science observation.
Often two are taken.  We will use both in this case and stack them later.

::

    dataselect ../playdata/example1/*.fits --tags ARC -o arcs.lis

.. telluric

A list for the science observations
-----------------------------------

In our case, the science observations can be selected from the observation
class, ``science``, that is how they are differentiated from the telluric
standards which are ``partnerCal``.

If we had multiple targets, we would need to split them into separate list. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../playdata/example1/*.fits --expr='observation_class=="science"' | showd -d object

    -----------------------------------------------------------------
    filename                                                   object
    -----------------------------------------------------------------
    ../playdata/example1/N20180106S0158.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0159.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0160.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0161.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0162.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0163.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0164.fits   2MASSI J0605019-234226
    ../playdata/example1/N20180106S0165.fits   2MASSI J0605019-234226

Here we only have one object from the same sequence.  If we had multiple
objects we could add the object name in the expression.

::

    dataselect ../playdata/example1/*.fits --expr='observation_class=="science" and object=="2MASSI J0605019-234226"' -o sci.lis

Bad Pixel Mask
==============
Starting with DRAGONS v3.1, the bad pixel masks (BPMs) are handled as
calibrations.  They are downloadable from the archive instead of being
packaged with the software. They are automatically associated like any other
calibrations.  This means that the user now must download the BPMs along with
the other calibrations and add the BPMs to the local calibration manager.

See :ref:`getBPM` in :ref:`tips_and_tricks` to learn about the various ways
to get the BPMs from the archive.

To add the static BPM included in the data package to the local calibration
database:

::

    caldb add ../playdata/example1/bpm*.fits


Master Flat Field
=================
GNIRS longslit flat fields are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.

.. todo:: discuss the odd-even problem.  probably with the interactive tool
   as people won't see that otherwise.

The flats will be stacked.

::

    reduce @flats.lis

The primitive ``normalizeFlat``, used in the recipe, has an interactive mode.
To activate the interactive mode:

::

    reduce @flats.lis -p interactive=True

The interactive tools are introduced in section :ref:`interactive`.

Processed Arc - Wavelength Solution
===================================
Obtaining the wavelength solution for GNIRS longslit data can be a complicated
topic.  The quality of the results and what to use depends greatly on the
wavelength regime and the grating.

Our observations are K-band, around 2.3 |um|, with the 111/mm grating.  We have
three options:

* In this configuration, the GCAL arc lamp observation has only 3 lines.
* The other option is to use the OH and |O2| sky emission lines in the science
  or telluric star data.  However, those lines are absent beyond ~2.3 |um|.
* The final option is to use telluric absorption in the science or telluric
  star data.

We will look at each of these 3 options and show what to look for and how to
come up with the best decision for this dataset, in this configuration.

Because the slit length does not cover the whole array, we want to know where
the unilluminated areas are location and ignore them when the distortion
correction is calculated (along the wavelength solution).  That information
is measured during the creation of the flat field and stored in the processed
flat.   Right now, the association rules do not automatically associate
flats to arcs, therefore we need to specify the processed flat on the
command line.  Using the flat is optional but it is recommended.

.. todo::  How about the zero adjustment?

Option 1 - The GCAL Arc Lamp
----------------------------
Two GNIRS longslit arcs were taken at the end of the science sequence.
We are going to use both and stack them to increase the signal in the weaker
of the 3 lines in that wavelength range.

::

    reduce @arcs.lis -p flatCorrect:flat=N20180106S0166_flat.fits

The primitive ``determineWavelengthSolution``, used in the recipe, has an
interactive mode. To activate the interactive mode:

::

    reduce @arcs.lis -p interactive=True flatCorrect:flat=N20180106S0166_flat.fits

The interactive tools are introduced in section :ref:`interactive`.


.. telluric

Science Observations
====================
The science target is cool brown dwarf.  The sequence is two successive ABBA
patterns.  DRAGONS will flatfield, subtract the sky, and stack the aligned
spectra, then extract the source.

.. note::  In this observation, there is only one source to extract.  If there
   were multiple sources in the slit, regardless of whether they are of
   interest to the program, DRAGONS will locate them, trace them, and extract
   them automatically. Each extracted spectrum is stored in an individual
   extension in the output multi-extension FITS file.

This is what one raw image looks like.

.. image:: _graphics/rawscience.png
   :width: 600
   :alt: raw science image

With all the calibrations in the local calibration manager, one only needs
to call |reduce| on the science frames to get an extracted spectrum.

::

    reduce @sci.lis

This produces a 2-D spectrum (``N20180106S0158_2D.fits``) which has been
flat fielded, wavelength-calibrated, corrected for distortion, sky-subtracted,
and stacked.  It also produces the 1-D spectrum (``N20180106S0158_1D.fits``)
extracted from that 2-D spectrum. The 1-D spectra are stored as 1-D FITS images
in extensions of the output Multi-Extension FITS file.

This is what the 2-D spectrum looks like.

::

    reduce -r display N20180106S0158_2D.fits

.. note::

    ``ds9`` must be launched by the user ahead of running the display primitive.
    (``ds9&`` on the terminal prompt.)

.. image:: _graphics/2Dspectrum.png
   :width: 600
   :alt: 2D stacked spectrum

The apertures found are listed in the log for the ``findApertures`` primitive,
just before the call to ``traceApertures``.  Information about the apertures
are also available in the header of each extracted spectrum: ``XTRACTED``,
``XTRACTLO``, ``XTRACTHI``, for aperture center, lower limit, and upper limit,
respectively.

This is what the 1-D flux-calibrated spectrum of our sole target looks like.

::

    dgsplot N20180106S0158_1D.fits 1

.. image:: _graphics/1Dspectrum.png
   :width: 600
   :alt: 1D spectrum

To learn how to plot a 1-D spectrum with matplotlib using the WCS from a Python
script, see Tips and Tricks :ref:`plot_1d`.

If you need an ascii representation of the spectum, you can use the primitive
``write1DSpectra`` to extract the values from the FITS file.

::

    reduce -r write1DSpectra N20180106S0158_1D.fits

The primitive outputs in the various formats offered by ``astropy.Table``.  To
see the list, use |showpars|.

::

    showpars N20180106S0158_1D.fits write1DSpectra

To use a different format, set the ``format`` parameters.

::

    reduce -r write1DSpectra -p format=ascii.ecsv extension='ecsv' N20180106S0158_1D.fits
