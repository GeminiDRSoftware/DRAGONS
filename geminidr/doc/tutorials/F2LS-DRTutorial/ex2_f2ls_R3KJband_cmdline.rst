.. ex2_f2ls_R3KJband_cmdline.rst

.. include:: symbols.txt

.. _f2ls_R3KJband_cmdline:

*************************************************************************
Example 2 - J-band R3K Longslit Point Source - Using the "reduce" command
*************************************************************************

We will reduce a F2 R3K 1.25 |um| longslit observation the recurrent nova
V1047 Cen using the "|reduce|" command that is operated directly from the
Unix shell  Just open a terminal and load the DRAGONS conda environment to
get started.

This observation uses the 2-pixel slit. The dither pattern is a standard
ABBA.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`f2ls_R3KJband_dataset`

Here is a copy of the table for quick reference.

+----------------------------+---------------------------------------------+
| Science                    || S20190702S0107-110                         |
+----------------------------+---------------------------------------------+
| Science darks (300s)       || S20190706S0431-437                         |
+----------------------------+---------------------------------------------+
| Science flat               || S20190702S0111                             |
+----------------------------+---------------------------------------------+
| Science flat darks (5s)    || S20190629S0029-035                         |
+----------------------------+---------------------------------------------+
| Science arc                || S20190702S0112                             |
+----------------------------+---------------------------------------------+
| Science arc darks (60s)    || S20190629S0085-091                         |
+----------------------------+---------------------------------------------+
| Science arc flat           || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric                   || S20190702S0099-102                         |
+----------------------------+---------------------------------------------+
| Telluric darks (25s)       || S20190706S0340-346                         |
+----------------------------+---------------------------------------------+
| Telluric flat              || Same as science flat                       |
+----------------------------+---------------------------------------------+
| Telluric flat darks (16s)  || Same as science flat darks                 |
+----------------------------+---------------------------------------------+
| Telluric arc               || S20190702S0103                             |
+----------------------------+---------------------------------------------+
| Telluric arc darks         || Same as science arc darks                  |
+----------------------------+---------------------------------------------+
| Telluric arc flat          || Same as telluric flat                      |
+----------------------------+---------------------------------------------+


Configuring the interactive interface
=====================================
In ``~/.dragons/``, add the following to the configuration file ``dragonsrc``::

    [interactive]
    browser = your_preferred_browser

The ``[interactive]`` section defines your preferred browser.  DRAGONS will open
the interactive tools using that browser.  The allowed strings are "**safari**",
"**chrome**", and "**firefox**".

Set up the Local Calibration Manager
====================================

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_cmdline`.

We recommend that you clean up your working directory (``playground``) and
start a fresh calibration database (``caldb init -w``) when you start a new
example.

Inspect and fix headers
=======================
It is unfortunately too common that the last frame of a science or
telluric sequence gets some, not all, of its header values from the next
(yes, future) frame which is normally, in the case of F2, a flat.  The
key headers to pay attention too are EXPTIME and LNRS.  They are both
associate with descriptors, so we will use "|showd|" to inspect the data.

First, navigate to the ``playground`` directory in the unpacked data package::

    cd <path>/f2ls_tutorial/playground

Let's inspect the ``exposure_time`` and the ``read_mode`` for the science
and the telluric data.  For a given sequence, all the values should match.

::

    dataselect ../playdata/example2/*.fits --xtags CAL --expr='observation_class=="science"' | showd -d exposure_time,read_mode
    dataselect ../playdata/example2/*.fits --xtags CAL --expr='observation_class!="science"' | showd -d exposure_time,read_mode

The science sequence::

    --------------------------------------------------------------------
    filename                                   exposure_time   read_mode
    --------------------------------------------------------------------
    ../playdata/example2/S20190702S0107.fits           300.0           8
    ../playdata/example2/S20190702S0108.fits           300.0           8
    ../playdata/example2/S20190702S0109.fits           300.0           8
    ../playdata/example2/S20190702S0110.fits           300.0           8

The telluric sequence::

    --------------------------------------------------------------------
    filename                                   exposure_time   read_mode
    --------------------------------------------------------------------
    ../playdata/example2/S20190702S0099.fits            25.0           1
    ../playdata/example2/S20190702S0100.fits            25.0           1
    ../playdata/example2/S20190702S0101.fits            25.0           1
    ../playdata/example2/S20190702S0102.fits            25.0           1


Everything looks good with all the exposure times and read modes matching.
Had there been discrepancies, you would have fixed them with ``fixheader``
as shown in Example 1 and Example 3.

Create file lists
=================
This data set contains science and calibration frames. For some programs, it
could contain different observed targets and different exposure times depending
on how you like to organize your playdata/example2 data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  However, DRAGONS provides tools to help you.

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata  and for a list
of |descriptors|.)

Make sure that you are in the ``playground`` directory of the unpacked
data package::

    cd <path>/f2ls_tutorial/playground

Several lists for the darks
---------------------------
The flats, the arcs, the telluric, and the science observations need
a master dark matching their exposure time.  We need a list of darks
for each set.   The exposure times are 5s for the flats, 60s for the arcs,
25s for the telluric frames, and 300s for the science frames.

::

    dataselect ../playdata/example2/*.fits --tags DARK --expr='exposure_time==5' -o dark5.lis
    dataselect ../playdata/example2/*.fits --tags DARK --expr='exposure_time==25' -o dark25.lis
    dataselect ../playdata/example2/*.fits --tags DARK --expr='exposure_time==60' -o dark60.lis
    dataselect ../playdata/example2/*.fits --tags DARK --expr='exposure_time==300' -o dark300.lis

One list for the flat
---------------------
Only one flat was obtained for this observation, one flat just after the
science sequence.  It can happen that a flat is also obtained after the
telluric sequence.  The recipe to make the master flats combines the input
flats if more than one is passed.  Therefore each flat group (one flat or
flats taken in succession) needs to be reduced independently.  Here, we just
need to send the filename of the unique flat to a list.

.. note:: No selection criteria are needed here since there is just one flat.
    Obviously, if your raw data directory contains all the data for the
    entire program you will have to apply selection criteria to ensure that
    the flats are sorted adequately.

::

    dataselect ../playdata/example2/*.fits --tags FLAT | showd -d ut_time

    ----------------------------------------------------------
    filename                                           ut_time
    ----------------------------------------------------------
    ../playdata/example2/S20190702S0111.fits   01:54:39.300000

::

    dataselect ../playdata/example2/*.fits --tags FLAT -o flat.lis

A list for the arcs
-------------------
There are two arcs.  One for the telluric sequence, one for the science
sequence.  The recipe to measure the wavelength solution will not stack the
arcs.  Therefore, we can conveniently create just one list with all the raw
arc observations in it and they will be processed independently.

::

    dataselect ../playdata/example2/*.fits --tags ARC -o arc.lis

A list for the telluric
-----------------------
DRAGONS does not recognize the telluric star as such.  This is because, at
Gemini, the observations are taken like science data and the Flamingos 2
headers do not
explicitly state that the observation is a telluric standard.  In most cases,
the ``observation_class`` descriptor can be used to differentiate the telluric
from the science observations, along with the rejection of the ``CAL`` tag to
reject flats and arcs.

::

    dataselect ../playdata/example2/*.fits --xtags CAL --expr='observation_class!="science"' -o tel.lis

A list for the science observations
-----------------------------------
The science observations can be selected from the observation
class, ``science``, that is how they are differentiated from the telluric
standards which are ``partnerCal``.

If we had multiple targets, we would need to split them into separate lists. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../playdata/example2/*.fits --xtags CAL --expr='observation_class=="science"' | showd -d object

    ----------------------------------------------------
    filename                                      object
    ----------------------------------------------------
    ../playdata/example2/S20190702S0107.fits   V1047 Cen
    ../playdata/example2/S20190702S0108.fits   V1047 Cen
    ../playdata/example2/S20190702S0109.fits   V1047 Cen
    ../playdata/example2/S20190702S0110.fits   V1047 Cen

Here we only have one object from the same sequence.  If we had multiple
objects we could add the object name in the expression.

::

    dataselect ../playdata/example2/*.fits --xtags CAL --expr='observation_class=="science" and object=="V1047 Cen"' -o sci.lis

Master Darks
============
Now that the lists are created, we just need to run |reduce| on each list.

::

    reduce @dark5.lis
    reduce @dark25.lis
    reduce @dark60.lis
    reduce @dark300.lis

Master Flat Fields
==================
Flamingos 2 longslit flat fields are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.

Flamingos 2 longslit master flat fields are created from the lamp-on flat(s)
and a master dark matching the flats exposure times.  Lamp-off flats are not
used.

In Flamingos 2 spectroscopic observations a blocking filter is used.  The
sharp drops in signal at both end makes fitting a function difficult. Our
recommendation is to set the region to be between the sharp drops and then
fit a low-order cubic spline.  This is a departure from what is being
recommended for the other Gemini spectrographs where a high-order is
recommended to fit all the wiggles.

For F2, only the overall shape should be fit.  The detailed fitting will be
taken care of when the sensitivity function is calculated using the telluric
standard star.

::

    reduce @flat.lis -p interactive=True   # using order 29 for now, full region.
   # also tried order 1 with region limited to the flat area.


<insert screenshot of the interactive fit>

We find that a region going from pixel 1090 to 1820 and an order of 1 is
leading to a reasonable fit.  Avoiding a fit that goes negative helps a lot.

Processed Arc - Wavelength Solution
===================================
Obtaining the wavelength solution for Flamingos 2 is fairly straightforward.
There are usually a sufficient number of lines in the lamp.

The recipe for the arc requires a flat as it contains a map of the
unilluminated areas.   The master dark is required because of the strong
pattern that is often horizontal and that could be interpreted as an emission
line if not removed.

The solution is normally found automatically, but it does not hurt to
visually inspect it in interactive mode.

::

    reduce @arc.lis -p interactive=True

Telluric Standard
=================
The telluric standard observed after the science observation is "hip 98805".
The spectral type of the star is A0.5V.

To properly calculate and fit a telluric model to the star, we need to know
its effective temperature.  To properly scale the sensitivity function (to
use the star as a spectrophotometric standard), we need to know the star's
magnitude.  Those are inputs to the ``fitTelluric`` primitive.

In Eric Mamajek's list "A Modern Mean Dwarf Stellar Color and Effective
Temperature Sequence"
(https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
the effective temperature of an A0.5V star as about 9500 K. The precise
value has only a small effect on the derived sensitivity and even less
effect on the telluric correction, so the temperature from any reliable
source can be used. Using Simbad, we find that the star has a magnitude
of J=7.498, which is the closest waveband to our observation.

Instead of typing the values on the command line, we will use a parameter file
to store them.  In a normal text file (here we name it "hip63036.param"),
we write::

    -p
    fitTelluric:bbtemp=9500
    fitTelluric:magnitude='J=7.498'

Then we can call the ``reduce`` command with the parameter file.  The telluric
fitting primitive can be run in interactive mode.

Note that the data are recognized by Astrodata as normal F2 longslit science
spectra.  To calculate the telluric correction, we need to specify the telluric
recipe (``-r reduceTelluric``), otherwise the default science reduction will be
run.

::

    reduce -r reduceTelluric @tel.lis @hip63036.param -p interactive=True prepare:bad_wcs=new

The ``prepare:bad_wcs=new`` is needed because the WCS in the raw data
is not quite right and that leads to an incorrect sky subtraction and
alignment.  See :ref:`badwcs` for more information.

<screenshot of the fit, maybe at the edges.  show that order 8 appears to be
the best compromise.>

# fitTelluric hits the LSF 0.5 limit.  Need to set order to 25 to get a decent corrected spectrum, and even then it
# kind of suck at the edges.
# fitTelluric plot is limited to the valid area.  But I believe that the model goes beyond and that's what I see
#   in telluricCorrect.
#
#  With order 1 flat, fitTelluric looks much better.  Order 8 does a nice job.

<screenshot of fit, and any relevant discussion.  Add when we are have
fixed the LSF problem.>

Science Observations
====================
The science target is recurrent nova.  The observation is one ABBA set.
DRAGONS will subtract the dark current, flatfield the data, apply the
wavelength calibration, subtract the sky, stack the aligned spectra.  Then the
source will be extracted to a 1D spectrum, the telluric features removed, and
the spectrum flux calibrated.

Following the wavelength calibration, the default recipe has an optional
step to adjust the wavelength zero point using the sky lines.  By default,
this step will NOT make any adjustment.  We found that in general, the
adjustment is so small as being in the noise.  If you wish to make an
adjustment, or try it out, see :ref:`wavzero` to learn how.

.. note::  When the algorithm detects multiple sources, all of them will be
     extracted.  Each extracted spectrum is stored in an individual extension
     in the output multi-extension FITS file.

This is what one raw image looks like.

<raw science image>

To run the reduction, call |reduce| on the science list.  The calibrations
will be automatically associated.  It is recommended to run the reduction
in interactive mode to allow inspection of and control over the critical
steps.

::

    reduce @sci.lis -p interactive=True prepare:bad_wcs=new findApertures:max_apertures=1 telluricCorrect:telluric=S20190702S0099_telluric.fits fluxCalibrate:standard=S20190702S0099_telluric.fits

<Until findApertures is fixed, use findApertures:max_apertures=1>
<Once we have telluric and specphot association, remove the manual assignment>

The exposure time of each of the four frames is 300 seconds.  The default time
interval for the sky subtraction association is 600 seconds.  The default
number of skies to use, ``min_skies`` is 2.  The routine will issues warnings
that it cannot find 2 sky frames compatible with the time interval.  The
default behavior in this case is to issue the warnings and ignore the time
interval constraint.  Here it works fine.  Depending on the sky conditions and
variability, another solution would be to set ``min_skies`` to 1 and always
catch the A or B frame closest in time.  Which works best for a given dataset
is something the users will have to judge for themselves.

<with bad LSF, at ``telluricCorrect`` ``-0.1`` shift is necessary.>

The 2D spectrum before extraction looks like this, with blue wavelengths at
the bottom and the red-end at the top.

<image of 2D stack>

The 1D extracted spectrum before telluric correction or flux calibration,
obtained with ``-p extractSpectra:write_outputs=True``, looks like this.

<image of _extracted>

The 1D extracted spectrum after telluric correction but before flux
calibration, obtained with ``-p telluricCorrect:write_outputs=True``, looks
like this.

<image of _telluricCorrected>

And the final spectrum, corrected for telluric features and flux calibrated.

::

    dgsplot S20190702S0107_1D.fits 1

<image of _1D>

# It appears that the "beyond filter cut off" signal is not being masked.  It shows up in science spectrum.
# telluricCorrect:  shows the masked pixel area.  It shouldn't.  Over the ** valid ** area, the model
#   looks ever so slightly better.
# dgsplot shows only the valid area for the extracted spectrum.
# For the _1D, it shows all the crap beyond.


