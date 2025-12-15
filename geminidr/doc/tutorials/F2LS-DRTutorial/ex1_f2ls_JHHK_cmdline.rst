.. ex1_f2ls_JHHK_cmdline.rst

.. include:: symbols.txt

.. _f2ls_JHHK_cmdline:

************************************************************************
Example 1 - JH and HK Longslit Point Source - Using the "reduce" command
************************************************************************

We will reduce a Flamingos 2 JH and a HK longslit observation of the 2022
eruption of the recurrent nova U Sco using the "|reduce|" command that is
operated directly from the Unix shell  Just open a terminal and load the
DRAGONS conda environment to get started.

The 2-pixel slit is used.  The dither sequence is ABBA-ABBA.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`f2ls_JHHK_dataset`

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

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="JH"' | showd -d exposure_time,read_mode
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="JH"' | showd -d exposure_time,read_mode

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="HK"' | showd -d exposure_time,read_mode
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="HK"' | showd -d exposure_time,read_mode

You will notice that the exposure times and read mode for each sequence match,
**except for the HK science sequence** where the last frame claims to have an
exposure time of 90 seconds instead of 25, and a read mode of 1 (LNRS keyword)
instead of 4.   Those are the values that apply to the next frame, the flat.
The data was taken with the correct exposure time and read mode, but the
headers are wrong.

::

    --------------------------------------------------------------------
    filename                                   exposure_time   read_mode
    --------------------------------------------------------------------
    ../playdata/example1/S20220617S0038.fits            25.0           4
    ../playdata/example1/S20220617S0039.fits            25.0           4
    ../playdata/example1/S20220617S0040.fits            25.0           4
    ../playdata/example1/S20220617S0041.fits            90.0           1

Let's fix that.  So that you can rerun these same commands before, we first
make a copy of the problematic file and give it a new name, leaving the
original untouched. Obviously, with your own data, you would just fix the
downloaded file once and for all, skipping the copy.  The tool ``fixheader``
changes the file in place.

::

    cp ../playdata/example1/S20220617S0041.fits ../playdata/example1/S20220617S0041_fixed.fits
    fixheader ../playdata/example1/S20220617S0041_fixed.fits EXPTIME 25
    fixheader ../playdata/example1/S20220617S0041_fixed.fits LNRS 4

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
|astrodatauser| for information about Astrodata  and for a list
of |descriptors|.)

Make sure that you are in the ``playground`` directory of the unpacked
data package::

    cd <path>/f2ls_tutorial/playground

Several lists for the darks
---------------------------
The flats, the arcs, the telluric, and the science observations need
a master dark matching their exposure time.  We need a list of darks
for each set, and for both JH and HK gratings.

::

    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==6' -o dark6.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==8' -o dark8.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==15' -o dark15.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==18' -o dark18.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==25' -o dark25.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==60' -o dark60.lis
    dataselect ../playdata/example1/*.fits --tags DARK --expr='exposure_time==90' -o dark90.lis

Four lists for the flats
------------------------
We have four observation sequences: science and telluric for both JH and HK
settings.  Each has its own flat.  The recipe to make the master
flats will combine the flats more than one is passed.  We need each flat to be
processed independently as they were taken at a slightly different telescope
orientation.  Therefore we need to separate them into four lists.

There are various ways to do that with |dataselect|.  Here use the name
of the disperser and a UT time selection.

We first check the times at which the flats were taken.  Then use that
information to set our selection criteria to separate them.

::

   dataselect ../playdata/example1/*.fits --tags FLAT | showd -d ut_time,disperser

::

    ----------------------------------------------------------------------
    filename                                           ut_time   disperser
    ----------------------------------------------------------------------
    ../playdata/example1/S20220617S0031.fits   00:30:30.100000          HK
    ../playdata/example1/S20220617S0042.fits   00:57:17.100000          HK
    ../playdata/example1/S20220617S0048.fits   01:06:37.100000          JH
    ../playdata/example1/S20220617S0077.fits   01:58:44.100000          JH

For HK, the telluric was taken before the science, for JH, it was taken after.
Therefore, we can construct our lists this way::

    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="JH" and ut_time<="01:56:00"' -o flatsciJH.lis
    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="JH" and ut_time>="01:56:00"' -o flattelJH.lis

    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="HK" and ut_time>="00:52:00"' -o flatsciHK.lis
    dataselect ../playdata/example1/*.fits --tags FLAT --expr='filter_name=="HK" and ut_time<="00:52:00"' -o flattelHK.lis

The exact UT time does not matter as long as it is between the two flats that
we want to separate.

A list for the arcs
-------------------
There are four arcs.  One for the telluric sequence, one for the science
sequence, and for both the JH and HK gratings.  The recipe to measure the
wavelength solution will not stack the arcs.  Therefore, we can conveniently
create just one list with all the raw arc observations in it and they will be
processed independently.

::

    dataselect ../playdata/example1/*.fits --tags ARC -o arc.lis

A list for the telluric
-----------------------
DRAGONS does not recognize the telluric star as such.  This is because, at
Gemini, the observations are taken like science data and the Flamingos 2
headers do not explicitly state that the observation is a telluric standard.
In most cases, the ``observation_class`` descriptor can be used to
differentiate the telluric from the science observations, along with the
rejection of the ``CAL`` tag to reject flats and arcs.

::

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="JH"' -o telJH.lis
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="partnerCal" and disperser=="HK"' -o telHK.lis

A list for the science observations
-----------------------------------
The science observations can be selected from the observation class,
``science``, that is how they are differentiated from the telluric standards
which are ``partnerCal``.

If we had multiple targets, we would need to split them into separate lists. To
inspect what we have we can use |dataselect| and |showd| together.

::

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science"' | showd -d object

::

    ---------------------------------------------------------
    filename                                           object
    ---------------------------------------------------------
    ../playdata/example1/S20220617S0038.fits         V* U Sco
    ../playdata/example1/S20220617S0039.fits         V* U Sco
    ../playdata/example1/S20220617S0040.fits         V* U Sco
    ../playdata/example1/S20220617S0041.fits         V* U Sco
    ../playdata/example1/S20220617S0041_fixed.fits   V* U Sco
    ../playdata/example1/S20220617S0044.fits         V* U Sco
    ../playdata/example1/S20220617S0045.fits         V* U Sco
    ../playdata/example1/S20220617S0046.fits         V* U Sco
    ../playdata/example1/S20220617S0047.fits         V* U Sco

Also, since we had to fix the exposure time for one of the files and we created
a copy instead of changing the original, we need to make sure only the science
frame with the correct exposure time of 25 seconds get picked up.  If you had
fixed the original, mostly likely what you will do with your data, you wouldn't
need to select on the exposure time.

::

    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="JH" and object=="V* U Sco"' -o sciJH.lis
    dataselect ../playdata/example1/*.fits --xtags CAL --expr='observation_class=="science" and disperser=="HK" and exposure_time==25 and object=="V* U Sco"' -o sciHK.lis

Master Darks
============
Now that the lists are created, we just need to run |reduce| on each list.

::

    reduce @dark6.lis
    reduce @dark8.lis
    reduce @dark15.lis
    reduce @dark18.lis
    reduce @dark25.lis
    reduce @dark60.lis
    reduce @dark90.lis

A bit of bash scripting can help by looping through the files::

    for file in $(ls dark*.lis); do reduce @$file; done

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
recommended for the other Gemini spectrograph where a high-order is
recommended to fit all the wiggles.

For F2, only the overall shape should be fit.  The detailed fitting will be
taken care of when the sensitivity function is calculated using the telluric
standard star.

Since both the science and the telluric flat are taken in the same
configuration, we can run the master flat recipe on one of them using the
interactive mode, then apply what we find to the other flat without going
into the interactive mode to save time.  Repeat for the other disperser.

Adjust the region in and the order to fit the overall shape while preventing
the fit from going negative (from pixel 1 to 2048).

For the JH flats::

    reduce @flatsciJH.lis -p interactive=True
    reduce @flattelJH.lis -p normalizeFlat:regions="395:1697" normalizeFlat:order=2

<screenshot of the JH flat fit>

For the HK flats::

    reduce @flatsciHK.lis -p interactive=True
    reduce @flattelHK.lis -p normalizeFlat:regions="280:1820" normalizeFlat:order=1

<screenshot of the HK flat fit>
<discussion to explain why 1.  To fit the shape, need to go beyond 6 and
then get flaring. 1 avoids flaring and negative fit>

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

Telluric Standards
==================
Two telluric standards are required for the reduction of this data set.  The
HK observations were done at the beginning of the program's sequence.  A
HK telluric was observed before the start of the science sequence.   The JH
observations were done at the end of the sequence and the matching JH telluric
was obtained afterwards.

The JH telluric is HIP 83920, a A0V star with an estimated temperature of
9700K and a H magnitude of 8.044.  The HK telluric is HIP 79156, a A0.5V star
with an estimated temperature of 9500K and a H magnitude of 7.576.

(Temperatures from Eric Mamajek's list "A Modern Mean Dwarf Stellar Color and
Effective Temperature Sequence"
https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
)

Those physical characteristic are required to properly calculate and fit a
telluric model to the star and scale the sensitivity function.  They are
fed to the primitive ``fitTelluric``.

Instead of typing the values on the command line, we will use a parameter file
to store them.  In normal text files (here we name the "hip83920.param" and
hip79156.param), we write for HIP 83920::

    -p
    fitTelluric:bbtemp=9700
    fitTelluric:magnitude='H=8.044'

and for HIP 79156::

    -p
    fitTelluric:bbtemp=9500
    fitTelluric:magnitude='H=7.576'

Then we can call the ``reduce`` command with the parameter file.  The telluric
fitting primitive can be run in interactive mode.

Note that the data are recognized by Astrodata as normal GNIRS longslit science
spectra.  To calculate the telluric correction, we need to specify the telluric
recipe (``-r reduceTelluric``), otherwise the default science reduction will be
run.

::

    reduce -r reduceTelluric @telJH.lis @hip83920.param -p interactive=True
    reduce -r reduceTelluric @telHK.lis @hip79156.param -p interactive=True prepare:bad_wcs=new

The WCS for the HK data are incorrect, hence the ``prepare:bad_wcs=new`` option.
See :ref:`badwcs` for more information.

Fits are bad.  LSF 0.5 limit.
JH spline3 30 with region: 865:1740  Not very good at the edges
HK spline3 30 with region: 1286:2503  Not very good at the edges

Science Observations
====================
The target is a recurrent nova, U Sco, that was going through an eruption at
the time of the observations.   The dither pattern is a standard ABBA, repeated
once.

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

This is what the raw images looks like, for JH and for HK.

<raw JH science image>
<raw HK science image>

To run the reduction, call |reduce| on the science list.  The calibrations
will be automatically associated.  It is recommended to run the reduction
in interactive mode to allow inspection of and control over the critical
steps.

::

    reduce @sciJH.lis -p interactive=True prepare:bad_wcs=new findApertures:max_apertures=1 extractSpectra:write_outputs=True telluricCorrect:write_outputs=True telluricCorrect:telluric=S20220617S0073_telluric.fits fluxCalibrate:standard=S20220617S0073_telluric.fits
    reduce @sciHK.lis -p interactive=True prepare:bad_wcs=new findApertures:max_apertures=1 extractSpectra:write_outputs=True telluricCorrect:write_outputs=True telluricCorrect:telluric=S20220617S0027_telluric.fits fluxCalibrate:standard=S20220617S0027_telluric.fits


JH:  Matches the publish spectrum except below 1.040 um where the continuum starts looking weird and
    likely exaggerated features show up.  This is the section where the fitTelluric fit struggles.

HK:  Right now, using the data rather than the model for telluricCorrect is far better.  The model leaves
    wiggles and the Pacshen alpha lines is very noisy.  Very good match to publish spectrum.  Again
    it's the blue end which is wrong, but not as wrong as for JH.

::

    reduce -r joinrecipe.joinSpectra *1D.fits -p scale=True

.. todo:: add that mini recipe to the F2 longslit recipe library.