.. ex1_ghost_stdonetarget_cmdline.rst

.. stdonetarget_cmdline:

***********************************************************************
Example 1 - Standard Resolution One Target - Using the "reduce" command
***********************************************************************

In this example we will reduce a GHOST observation of the star XX Oph using the
"|reduce|" command that is operated directly from the unix shell. Just open a
terminal and load the DRAGONS conda environment to get started.

This observation uses IFU-1 for the target.  IFU-2 is stowed.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`datastdonetarget`.

Here is a copy of the table for quick reference.

+-----------------+-------------------------------------------------+
| Science         || S20230416S0079 (blue:2x2,slow; red:2x2,medium) |
+-----------------+-------------------------------------------------+
| Science biases  || S20230417S0011-015                             |
+-----------------+-------------------------------------------------+
| Science Flats   || S20230416S0047 (1x1; blue:slow; red:medium)    |
+-----------------+-------------------------------------------------+
| Science Arcs    || S20230416S0049-51 (1x1)                        |
+-----------------+-------------------------------------------------+
| Flats Biases    || S20230417S0036-40 (1x1; blue:slow; red:medium) |
+-----------------+                                                 |
| Arc Biases      ||                                                |
+-----------------+-------------------------------------------------+
| Standard        || S20230416S0073 (blue:2x2,slow; red:2x2,medium) |
| (CD -32 9927)   ||                                                |
+-----------------+-------------------------------------------------+
| Standard biases || In this case, the calibrations for the         |
+-----------------+  science can be used for the standard star.     |
| Standard flats  ||                                                |
+-----------------+                                                 |
| Standard arc    ||                                                |
+-----------------+                                                 |
| Std flat biases ||                                                |
+-----------------+                                                 |
| Std arc biases  ||                                                |
+-----------------+-------------------------------------------------+

Special Step for Engineering Data
=================================
Because the data in this tutorial was obtained during commissioning, they
are identified as "engineering" data.  DRAGONS refuses to use such data, as
a safeguard.  To use the data anyway, we need to modify the program ID and
make the data look non-engineering.  We run the following script to do that.

It is unclear at this time if this will be applicable to the SV data.

::

  cd <path>/ghost_tutorial
  python fixprogid.py playdata/example1/*.fits


Set up the Local Calibration Manager
====================================

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_cmdline`.

Save some typing
================
Because the GHOST data reduction package is not yet fully integrated into
DRAGONS, the tools need to be told to use it.  This can be done with options
to the tools.   To save some typing, we can create aliases.

::

   alias gdataselect="dataselect --adpkg=ghost_instruments"
   alias gshowd="showd --adpkg=ghost_instruments"
   alias greduce="reduce --adpkg=ghost_instruments --drpkg=ghostdr"
   alias gshowpars="showpars --adpkg=ghost_instruments --drpkg=ghostdr"

The Files
=========
Unlike for other Gemini instruments, the GHOST raw data are "bundles".  They
contain multiple exposures from the red channel, multiple exposures for the
blue channel, and multiple slit-viewer images.

To keep our work directory clean, at least while learning how to reduce
GHOST data, we will de-bundle the files we need as we need them and create
list of data to reduce as we need them.

It might be tempting to de-bundle all the data at once, but beware of memory
issues.  GHOST raw bundles are very large.  You will also be drowned in files.

Let's inspect the data.  (It take a little long to run, the bundle files are
large.)

::

  cd <path>/ghost_tutorial/playground
  gshowd ../playdata/example1/*.fits -d object,detector_x_bin,detector_y_bin,read_mode

::

    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    filename                                        object                      detector_x_bin                      detector_y_bin                                                read_mode
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ../playdata/example1/S20230416S0047.fits      GCALflat   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230416S0049.fits          ThAr   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230416S0050.fits          ThAr   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230416S0051.fits          ThAr   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230416S0073.fits   CD -32 9927   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230416S0079.fits        XX Oph   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0011.fits          Bias   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0012.fits          Bias   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0013.fits          Bias   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0014.fits          Bias   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0015.fits          Bias   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 2, 'red': 2, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0036.fits          Bias   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0037.fits          Bias   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0038.fits          Bias   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0039.fits          Bias   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}
    ../playdata/example1/S20230417S0040.fits          Bias   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 1, 'red': 1, 'slitv': 2}   {'blue': 'slow', 'red': 'medium', 'slitv': 'standard'}

Master Biases
=============
In this section, we will create all the master biases that we need.  Here is
the list of biases we need to produce:

* A bias for slit-viewer camera
* A bias for science and standard, red channel
* A bias for science and standard, blue channel
* A bias for flat and arc, red channel
* A bias for flat and arc, blue channel

The biases must match the binning and read-mode of the data they will be used
on.  The binning of the flats and arcs is always 1x1.  While the read-mode
for the flats must match the science, there is no such requirement for the
arcs.  If the arcs have a different read-mode from the science and flats, you
will need an extra set of biases for the arc.   Fortunately, this is
not needed here since all the data was obtained with the same read-mode for
all red and all blue exposures.

Debundle the biases
-------------------

::

  gdataselect ../playdata/example1/*.fits --tags BIAS -o biasbundles.lis

  greduce @biasbundles.lis

.. note::  The GHOST data reduction software currently depends on `pysynphot`.
    That package issues an annoying but completely harmless `UserWarning`.
    What it complains about is not used in the GHOST software.
    Just ignore it until we have time to clean it up.

    ::

        /Users/klabrie/condaenvs/ghost3.0.4/lib/python3.7/site-packages/pysynphot/locations.py:46: UserWarning: PYSYN_CDBS is undefined; functionality will be SEVERELY crippled.
          warnings.warn("PYSYN_CDBS is undefined; functionality will be SEVERELY "
        /Users/klabrie/condaenvs/ghost3.0.4/lib/python3.7/site-packages/pysynphot/locations.py:345: UserWarning: Extinction files not found in extinction
          warnings.warn('Extinction files not found in %s' % (extdir, ))


Reduce the slit biases
----------------------
All the slit biases, regardless of binning or read mode in the blue and red
channels, are identical.  Then can all be stacked together to reduce noise.

::

  gdataselect *.fits --tags BIAS,SLIT -o biasslit.lis

  greduce @biasslit.lis

Reduce the science biases
-------------------------

::

  gdataselect *.fits --tags BIAS,RED \
    --expr="detector_x_bin==2 and detector_y_bin==2" -o biasredsci.lis
  gdataselect *.fits --tags BIAS,BLUE \
    --expr="detector_x_bin==2 and detector_y_bin==2" -o biasbluesci.lis

  greduce @biasredsci.lis
  greduce @biasbluesci.lis

All the data was obtained with the same read modes.  If this is not the case
for your data and you need to select on read mode, use an expression like
this one::

  --expr="detector_x_bin==2 and detector_y_bin==2 and read_mode=='slow'"

.. note::  You may see the following error message::

       ERROR - ValueError: zero-size array to reduction operation minimum which has no identity

    If so, your bias frame is corrupted (all pixels have the same value)
    and you should find an alternative bias with the same binning and
    read speeds in the archive and use that instead.


Reduce the flat/arc biases
--------------------------
The flats and the arcs were taken in the same read mode.  Therefore, we can
use the same set of biases for the flats and the arcs.  If they had been
observed in different read modes, you would need a set for the flats and a
set for the arcs.  Fortunately, not the case here, one set for both.

::

  gdataselect *.fits --tags BIAS,RED \
    --expr="detector_x_bin==1 and detector_y_bin==1" -o biasredflatarc.lis
  gdataselect *.fits --tags BIAS,BLUE \
    --expr="detector_x_bin==1 and detector_y_bin==1" -o biasblueflatarc.lis

  greduce @biasredflatarc.lis
  greduce @biasblueflatarc.lis


Master biases to Calibration Database
-------------------------------------
The output master biases, like all ``reduce`` products, are written to disk in
the work directory, ie. where ``reduce`` was called.  For calibrations, the final
calibration files are also written in the ``calibrations`` directory, in a
subdirectory representing the type of calibrations.  For the biases,
``calibrations/processed_bias/``.

This is a safe copy of the calibrations that will be needed later allowing
us the freedom to clean the work directory between steps, which is
particularly helpful in the case of GHOST.

Since we will indeed clean up the work directory, we will add the safe files in
``calibrations`` to the calibration manager database instead of the files in the
work directory.  A reminder that the files are not added to the database, only
the information about them and their location on disk; if you delete the file
on disk it is gone even if information about it remains in the database.

::

  caldb add calibrations/processed_bias/*.fits

Clean up
--------
GHOST reduction creates a lot of, often big, files in the work directory.  It
is recommended to clean up between each reduction phase.  If you want to save
the intermediate files, move them (``mv``) somewhere else at least.  In this
tutorial, we will simply delete them.

::

  rm *.fits


Master Flats and Slit-flats
===========================

Debundle Flats
--------------

::

  gdataselect ../playdata/example1/*.fits --tags FLAT -o flatbundles.lis

  greduce @flatbundles.lis

Reduce the Slit-flat
--------------------
The slit-flat is required to reduce the red and blue channel flats, so it is
important to reduce it first and add it to the calibration database.

::

  gdataselect *.fits --tags SLITFLAT -o slitflat.lis

  greduce @slitflat.lis
  caldb add calibrations/processed_slitflat/*.fits

.. note::  You will see this message in the logs::

       ERROR - Inputs have different numbers of SCI extensions.

    You can safely ignore it.  It is expected and the wording is misleading.
    This is not an real error.

Reduce the Flats
----------------
The flats have a 1x1 binning and must match the read mode of the science
data.  If the science data is binned, the software will bin the 1x1 flats
to match.

::

  gdataselect *.fits --tags FLAT,RED -o flatred.lis
  gdataselect *.fits --tags FLAT,BLUE -o flatblue.lis

  greduce @flatred.lis
  greduce @flatblue.lis
  caldb add calibrations/processed_flat/*.fits

.. note::  If you are reducing out-of-focus data from the December 2023 FT run,
    you should add the flag::

       -p smoothing=6

    when reducing the flats (not the slit-flat). The value of 6 (the FWHM in
    pixels of the Gaussian smoothing kernel) applied to the slit-viewer
    camera images (which are in focus) seems to work well but may not be
    optimal. The value is stored in the header of the processed flat so it
    is applied automatically to the reduction of the arc and on-sky frames
    that use the flat. You are welcome to try other values.

Clean up
--------
With the calibrations safely in the ``calibrations`` directory, we can clean
the work directory

::

    rm *.fits

Arcs
====
The arcs have a 1x1 binning, the read mode does not matter.  It does save
processing if they are of the same read mode as the flats as different biases
for the specific read mode are in such case not needed.  If the science data is binned,
the software will bin the 1x1 arcs to match.

Debundle the Arcs
-----------------

::

  gdataselect ../playdata/example1/*.fits --tags ARC -o arcbundles.lis

  greduce @arcbundles.lis

Reduce the slit-view data
-------------------------
We have 3 slit images for the arc but we really just need one.  We grab
the first one.

::

  gdataselect *.fits --tags ARC,SLIT | head -n 1 > arcslit.lis

  greduce @arcslit.lis
  caldb add calibrations/processed_slit/*.fits

Reduce the arcs
---------------

::

  gdataselect *.fits --tags ARC,RED -o arcred.lis
  gdataselect *.fits --tags ARC,BLUE -o arcblue.lis

  greduce @arcred.lis
  greduce @arcblue.lis
  caldb add calibrations/processed_arc/*.fits

.. note::  If you want to save a plot of the wavelength fits,
    add ``-p fitWavelength:plot1d=True`` to the ``greduce`` call.
    A PDF will be created.

Clean up
--------
With the calibrations safely in the ``calibrations`` directory, we can clean
the work directory

::

    rm *.fits

Spectrophotometric Standard
===========================
Unlike for GMOS, the standards are not automatically recognized as such.
This is something that has not been implemented at this time.
Therefore to select them, we will need to use the object's name.

Debundle the Standard
---------------------

::

  gdataselect ../playdata/example1/*.fits --expr="object=='CD -32 9927'" -o stdbundles.lis

  greduce @stdbundles.lis

Reduce the slit-view data
-------------------------
Since we have cleaned up all the intermediate files as we went along, we
are able to just select on the tag SLIT.  If we had not cleaned up, we would
need to use the object name like we did above.

::

  gdataselect *.fits --tags SLIT -o stdslit.lis

  greduce @stdslit.lis
  caldb add calibrations/processed_slit/S20230416S0073_slit_*.fits

Reduce the standard star
------------------------
Since we have cleaned up all the intermediate files as we went along, we
are able to just select on the tag RED and BLUE.  If we had not cleaned up,
we would need to use the object name like we did above for the bundle.

This step takes a while.

::

  gdataselect *.fits --tags RED -o stdred.lis
  gdataselect *.fits --tags BLUE -o stdblue.lis

  greduce -r reduceStandard @stdred.lis
  greduce -r reduceStandard @stdblue.lis

The reduced spectrophotometric standard observations are the ``_standard``
files.

The ``_flatBPMApplied`` files are the last 2D images of the spectra
before it gets extracted to 1D.  They are saved just in case you want to
inspect them.  They are not used for further reduction.

For the wavelength calibration, the pipeline will try to find an arc taken
before the observation and one taken after.  If it finds two, it will use them
both, however, one is enough.  This is what happens here: the software finds
a "before" arc, but no "after" arc.  So, do not be alarmed by the messages
saying that it failed to find an arc, it's okay, it got one, it's enough.

This standard observation has three red arm exposures.  They are not stacked.
They possibly should but the software was delivered without a final stacking
capability.  So for now, no stacking.  In this case, one exposure is bright
enough, and there's no problem at all using only one of the red exposures.
Pick one, anyone if the conditions were stable, they should all look the same.
If you suspect that the conditions were highly variable, you can inspect the
``_flatBPMApplied`` files and see which one is the brightest and use that one.
To display such a file, launch ``ds9`` and type::

   greduce -r display -p zscale=False S20230416S0073_red001_flatBPMApplied.fits

Since here it doesn't matter which file we use, we pick ``red001``.

Clean up
--------
Because the processed standard files are not recognized as such they are **NOT**
copied to the ``calibrations`` directory.  So you have to be very careful here
with your clean up.

You want to keep the `_standard` files for sure, those are the
calibration files that will be used to reduce the standard.  You might want to
keep the `_flatBPMApplied` files for visualization purposes.

The selective clean-up looks like this.

::

  rm *_slit.fits
  rm *_red???.fits
  rm *_blue???.fits

Science Frames
==============
As explained above, unlike for GMOS, the standards are not automatically
recognized as such. They are just like any other science observation.
Therefore to select the science, we will need to use the object's name.

Debundle the Science Frames
---------------------------

::

  gdataselect ../playdata/example1/*.fits --expr="object=='XX Oph'" -o scibundles.lis

  greduce @scibundles.lis

Reduce the slit-view data
-------------------------

::

  gdataselect *.fits --tags SLIT -o scislit.lis

  greduce @scislit.lis
  caldb add calibrations/processed_slit/S20230416S0079*.fits

Reduce the Science Frames
-------------------------
The processed standards are not associated automatically for GHOST.
They need to be specified on the command line with the ``-p`` flag.

The standard we are using is "CD -32 9927".  This is one of the baseline
standards Gemini uses.  The flux profile of that star is available in
DRAGONS and the name will be recognized and the file automatically retrieved.

If you were to use a spectrophotometric standard not on the Gemini list, you
would need to provide that flux standard file with the
``-p specphot_file=path/name_of_file``.  The accepted format are the "IRAF
format" and the HST calspec format.

.. note::  Possible customizations.

   * The sky subtraction can be turned off with ``-p extractProfile:sky_subtract=False``
     if it is found to add noise.
   * If you expected IFU-2 to be on-sky but there's an accidental source, tell
     the software that there is a source and it isn't sky with
     ``-p extractProfile:ifu2=object``.
   * If you do not want the barycentric correction, turn is off with
     ``-p barycentricCorrect:correction_factor=1``.

.. warning:: The reduction of the red channel is very slow.  Launch
  and go get a coffee or something.  Make sure that you got the name for the
  standard star file correct or it will crash after having done most of the
  work and you will have to start again.  Not fun.  The blue channel in this
  example reduces rather quickly.

::

  gdataselect *.fits --tags RED --expr="object=='XX Oph'" -o scired.lis
  gdataselect *.fits --tags BLUE --expr="object=='XX Oph'" -o sciblue.lis

  greduce @scired.lis -p standard=S20230416S0073_red001_standard.fits
  greduce @sciblue.lis -p standard=S20230416S0073_blue001_standard.fits

  dgsplot S20230416S0079_red001_dragons.fits 1 --bokeh

The final products are the ``_dragons`` files.  In those files, all the orders
have been stitched together with the wavelength on a log-linear scale,
calibrated to in-air wavelengths and corrected for barycentric motion (unless
that correction is turned off.)

The first extension (the "1" in the call to ``dgsplot`` above) is the spectrum.
The second extension is the spectrum of the sky.  This is for an observation
with one object and sky subtraction turned on (default).  Here's the list of
possible configurations:

* One object, sky subtraction: 2 spectra per order: sky-subtracted object
  spectrum, then sky spectrum
* Two objects, sky subtraction: 3 spectra per order: sky-subtracted object1
  spectrum, sky subtracted object2 spectrum, sky spectrum
* One object, no sky subtraction: 1 spectrum per order: object spectrum
* Two objects, no sky subtraction: 2 spectra per order: object1 spectrum,
  object2 spectrum


.. note::  If you are reducing standard-resolution out-of-focus data from
    the December 2023 FTrun in two-object mode or with one of the IFUs stowed
    you may see "ripple" artifacts in your data due to contamination of the
    sky fibres by light from the target(s). Using
    ``-p extractProfile:sky_subtract=False weighting=uniform`` may help.


It is possible to write the spectra to a text file with ``write1DSpectrum``,
for example::

  greduce -r write1DSpectra S20230416S0079_red001_dragons.fits

The primitive outputs in various formats offered by ``astropy.Table``.  To see
the list, use |showpars|.

::

  gshowpars S20230416S0079_red001_dragons.fits write1DSpectra

The ``_dragons`` files are probably what most people will want to use for
making their measurements.

The files ``_calibrated`` are the reduced spectra *before* the stitching
the orders and the format of the file is more complex and somewhat less
accessible.  But if you need it, you have it.  The flux pixels are in a
3D array with the first axis of size 2, one for target, one for sky, then a
second axis being the wavelength direction, and finally a third axis of
30-something orders.  The ``WAVL`` extension contains the wavelength at each
of the pixels in the wavelength-order array.

While we have three red exposures, the software does not stack them.  The
software was delivered without stacking.  If the sky conditions (cloud, seeing)
were stable and the individual spectra do not have large numbers of cosmic
rays, it is possible to use DRAGONS' ``stackFrames`` with no rejection, no
scaling to create a stacked spectrum.  Please use your best judgement.

::

    greduce -r stackFrames -p reject_method=none *red???_dragons.fits


