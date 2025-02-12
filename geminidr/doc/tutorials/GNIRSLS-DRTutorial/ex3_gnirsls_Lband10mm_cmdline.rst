.. ex3_gnirsls_Lband10mm_cmdline.rst

.. include:: symbols.txt

.. _gnirsls_Lband10mm_cmdline:

**************************************************************************
Example 3 - L-band Longslit Point Source - Using the "reduce" command line
**************************************************************************

We will reduce the GNIRS L-band longslit observation of "HD41335", a Be-star,
using the "|reduce|" command that is operated directly from the unix shell.
Just open a terminal and load the DRAGONS conda environment to get started.

This observation uses the 10 l/mm grating, the long-red camera, a 0.1 arcsec
slit, and is centered at 3.7 |um|.  The dither pattern is a standard
ABBA sequence.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`gnirsls_Lband10mm_dataset`

Here is a copy of the table for quick reference.

+---------------------+----------------------------------------------+
| Science             || N20180114S0121-124                          |
+---------------------+----------------------------------------------+
| Science flats       || N20180114S0125-132                          |
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
have to do it.  However, DRAGONS provides tools to help you.

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata.)

First, navigate to the ``playground`` directory in the unpacked data package::

    cd <path>/gnirsls_tutorial/playground

A list for the flats
--------------------
The GNRIS flats will be stack together.  Therefore it is important to ensure
that the flats in the list are compatible with each other.  You can use
`dataselect` to narrow down the selection as required.  Here, we have only
the flats that were taken with the science and we do not need extra selection
criteria.

::

    dataselect ../playdata/example3/*.fits --tags FLAT -o flats.lis

A list for the science observations
-----------------------------------

In our case, the science observations can be selected from the observation
class, ``science``, that is how they are differentiated from the telluric
standards which are ``partnerCal``.

If we had multiple targets, we would need to split them into separate lists. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../playdata/example3/*.fits --expr='observation_class=="science"' | showd -d object

    --------------------------------------------------
    filename                                    object
    --------------------------------------------------
    ../playdata/example3/N20180114S0121.fits   HD41335
    ../playdata/example3/N20180114S0122.fits   HD41335
    ../playdata/example3/N20180114S0123.fits   HD41335
    ../playdata/example3/N20180114S0124.fits   HD41335

Here we only have one object from the same sequence.  If we had multiple
objects we could add the object name in the expression.

::

    dataselect ../playdata/example3/*.fits --expr='observation_class=="science" and object=="HD41335"' -o sci.lis

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

    caldb add ../playdata/example3/bpm*.fits

Master Flat Field
=================
GNIRS longslit flat field are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.

The GNIRS longslit flatfield requires only lamp-on flats.  Subtracting darks
only increases the noise.

The flats will be stacked.

::

    reduce @flats.lis

GNIRS data is affected by a "odd-even" effect where alternate rows in the
GNIRS science array have gains that differ by approximately 10 percent.  When
you run ``normalizeFlat`` in interactive mode you can clearly see the two
levels.

In interactive mode, the objective is to get a fit that falls inbetween the
two sets of points, with a symmetrical residual fit.  In this case, because
of the rapid variations around pixel 800, increasing the order could improve
the final results.  Setting ``order=50`` fits that area well while still
offering a good fit elsewhere.

Note that you are not required to run in interactive mode, but you might want
to if flat fielding is critical to your program.

::

    reduce @flats.lis -p interactive=True

The interactive tools are introduced in section :ref:`interactive`.

.. image:: _graphics/gnirsls_Lband10mm_evenoddflat.png
   :width: 600
   :alt: Even-odd effect in flats

Processed Arc - Wavelength Solution
===================================
The wavelength solution for L-band and M-band data is derived from the telluric
emission lines in the science frames.  The quality of the wavelength solution
depends on the resolution and brightness of the telluric lines.

Wavelength calibration from sky lines is better done in interactive mode
despite our efforts to automate the process.

To use the sky lines in the science frames, we invoke the
``makeWavecalFromSkyEmission`` recipe.

::

    reduce @sci.lis -r makeWavecalFromSkyEmission -p interactive=True

It is very important to inspect the line identification.  Using the defaults,
like we did above, careful inspection shows that the line identification is
wrong.  Zooming in, we see the result below, not how the lines do not align.

.. image:: _graphics/gnirsls_Lband10mm_wrongarcID.png
   :width: 600
   :alt: Incorrect arc line identifications

We get a good fit by changing the "Minimum SNR for peak detection" value
to 5 in the panel on the left and then clicking the "Reconstruct points" button.

.. image:: _graphics/gnirsls_Lband10mm_correctarcID.png
   :width: 600
   :alt: Correct arc line identifications


.. note:: It is possible to set the minimum SNR from the command line by
   adding ``-p determineWavelengthSolution:min_snr=5`` to the ``reduce`` call)


.. telluric

Science Observations
====================
The science target is a Be star.  The sequence is one ABBA dither pattern.
DRAGONS will flatfield, wavelength calibrate, subtract the sky, stack the
aligned spectra, and finally extract the source.

Note that at this time, DRAGONS does not offer tools to do the telluric
correction and flux calibration.  We are working on it.

This is what one raw image looks like.

.. image:: _graphics/gnirsls_Lband10mm_raw.png
   :width: 400
   :alt: raw science image

With all the calibrations in the local calibration manager, simply call
|reduce| on the science frames to get an extracted spectrum.

::

    reduce @sci.lis

To run the reduction with all the interactive tools activated, set the
``interactive`` parameter to ``True``.

::

    reduce @sci.lis -p interactive=True

The default fits are all good, though the trace can be improved by setting
the order to 5.

The final 2D spectrum and the extracted 1D spectrum are shown below.

.. image:: _graphics/gnirsls_Lband10mm_2d.png
   :width: 400
   :alt: 2D spectrum

.. image:: _graphics/gnirsls_Lband10mm_1d.png
   :width: 600
   :alt: 1D extracted spectrum
