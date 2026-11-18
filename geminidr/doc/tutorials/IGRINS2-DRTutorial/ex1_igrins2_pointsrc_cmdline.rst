.. ex1_igrins2_pointsrc_cmdline.rst

.. include:: symbols.txt

.. _igrins2_pointsrc_cmdline:

*************************************************
Example 1 - Point Source - Command Line Reduction
*************************************************

In this example, we will reduce an IGRINS-2 observation of a CEMP-no star
using the "|reduce|" command that operates directly from the unix shell.
Just open a terminal and load the DRAGONS environment to get started.

The dither pattern is a standard ABBA sequence.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`igrins2_pointsrc_dataset`

Here is a copy of the table for quick reference.

+--------------------------------+-----------------------------------+
| Science                        || N20260130S0179-182               |
+--------------------------------+-----------------------------------+
| Flats                          || N20260130S0369,373,377,382,385,  |
|                                || 388,391,396,401,407-427          |
+--------------------------------+-----------------------------------+
| Sky for wavelength calibration || N20260130S0021                   |
+--------------------------------+-----------------------------------+
| Telluric                       || N20260130S0183-186               |
+--------------------------------+-----------------------------------+

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

Create file lists
=================
This dataset contains science and calibration files for one target.  For some
programs, it could contain different targets and standards depending on how
you organize your data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  However, DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps.
It uses Astrodata tags and |descriptors| to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata and for a list
of |descriptors|.)

First, navigate to the ``playground`` directory in the unpacked data package::

    cd <path>/igrins2_tutorial/playground


Unbundle the Bundles
--------------------
The IGRINS-2 data raw data files are organized in bundles that contain the
data from the H-band arm and the K-band arm.  The first step is to debundle
the data so that the H-band and K-band files are separated.

::

    dataselect ../playdata/example1/*.fits --tags BUNDLE -o allbundles.lis
    reduce @allbundles.lis

Two lists for the flats
-----------------------
The IGRINS-2 flats consist of a series of lamp-on and lamp-off exposures.
For this observation, there are 20 lamp-on and 10 lamp-off exposures.  The
software knows how to combine the lamp-on and lamp-off exposures to create
the master flat.  For each arm, we will put all the flats associated with
the science sequence in one list.

::

    dataselect *.fits --tags FLAT --expr='band=="H"' -o flatH.lis
    dataselect *.fits --tags FLAT --expr='band=="K"' -o flatK.lis

Two lists for the zenith sky frame
----------------------------------
The zenith sky frame is used to create the wavelength solution using the
sky emission lines instead of the arc lamp lines.  The zenith sky frame is
generally a single exposure.

::

    dataselect *.fits --tags SKY --expr='band=="H"' -o skywlenH.lis
    dataselect *.fits --tags SKY --expr='band=="K"' -o skywlenK.lis


Two lists for the telluric
--------------------------
The IGRINS-2 telluric standards are tagged STANDARD and can be easily separated
from the science observations.

::

    dataselect *.fits --tags STANDARD --expr='band=="H"' -o telluricH.lis
    dataselect *.fits --tags STANDARD --expr='band=="K"' -o telluricK.lis


Two lists for the science observations
--------------------------------------
The science observations are everything not a calibration, ie. not tagged
``CAL``.

::

    dataselect *.fits --xtags CAL --expr='band=="H"' -o scienceH.lis
    dataselect *.fits --xtags CAL --expr='band=="K"' -o scienceK.lis


Master Flat Field
=================
The IGRINS-2 flats are obtained at the end of the night.  They are a series
of lamp-on and lamp-off exposures.  The software will combine the
lamp-on and lamp-off exposures and create the master flat.

::

    reduce @flatH.lis -p normalizeFlat:niter=3
    reduce @flatK.lis -p normalizeFlat:niter=3

While the non-interactive solution should be correct, if you wish to inspect
the fit run the flat normalization in interactive mode by add
`-p interactive=True` to the command.

::

    reduce @flatH.lis -p interactive=True normalizeFlat:niter=3

.. .. image:: _graphics/igrins2_interflat.png
..   :width: 600
..   :alt: Interactive flat field inspection.

.. todo:: Issues with the interactive tool and the fit.  This is using
   branch tutorial/from_feature_igrins2.  The tabs for each order are not
   working properly.  There's one unreadable tab and a scrollbar.  As for the
   fit, it is too low, it is being dragged down bye the absorption lines.
   What are those absorption lines anyway?

   remove the ``normalizeFlat:niter=3`` from the command once the defaults
   have been adjusted.  Bill also finds that niter=3 works better than niter=1.


Wavelength Solution
===================
The wavelength solution is derived from the zenith sky frame.  The sky
emission lines are used to derive the wavelength solution.  The solution is
a 2D fit of all the sky lines in all the orders.

.. note:: The telluric standard will later be used to refine the wavelength
   solution using the telluric absorption.

::

    reduce @skywlenH.lis
    reduce @skywlenK.lis


Telluric Standard
=================
The telluric standard observed after the science observations is "HD 97585".
The spectral type of the star is A0V.

To properly calculate and fit a telluric model to the star, we need to know
its effective temperature.  To properly scale the sensitivity function (to
use the star as a spectrophotometric standard), we need to know the star's
magnitude.  Those are inputs to the ``fitTelluric`` primitive.

The software will automatically query the SIMBAD database to get the star's
spectral type and magnitude.  Then it will use the spectral type to derive the
effective temperature from a Eric Mamajek's list "A Modern Mean Dwarf Stellar
Color and Effective Temperature Sequence"
(https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt).

To reduce the telluric standard:

::

    reduce @telluricH.lis
    reduce @telluricK.lis

You can inspect the telluric fit in interactive mode by adding
`-p interactive=True` to the command.

::

    reduce @telluricH.lis -p interactive=True

The output processed telluric standard will contain the telluric model and the
sensitivity function, as well as a refined wavelength solution derived using
the telluric features.

.. note:: If you wish to override the automatically derived values, or if you
   do not have an internet connection, you can set
   the ``fitTelluric`` primitive parameters yourself.  For example, in a normal
   text file (say "hd97585_H.param"), we write::

        -p
        fitTelluric:bbtemp=9700
        fitTelluric:magnitude='H=5.391'

   For K-band, in this case the magnitude would be set to K=5.333.

   This parameter file can then be added to the command line with
   ``@hd97585_H.param``::

        reduce @telluricH.lis @hd97585_H.param


Science Observations
====================
The science target is a CEMP-no star.  The observations were taken in the
standard ABBA dither pattern.  The software will automatically combine the
dithered exposures.

The reduction steps include flat fielding, wavelength calibration, telluric
correction, and flux calibration.  The final output will be a fully reduced
spectrum of the science target.

This is what one raw image looks like.

.. .. image:: _graphics/gnirsxd_SXD32mm_raw.png
..   :width: 400
..   :alt: raw science image

.. what you see in that image are the x orders from x at the bottom to x at the
   top.

With all the calibrations in the local calibration manager, one only needs
to call |reduce| on the science frames to get an extracted spectrum.

::

    reduce @scienceH.lis
    reduce @scienceK.lis

To run the reduction with all the interactive tools activated, set the
``interactive`` parameter to ``True``.

::

    reduce @scienceH.lis -p interactive=True

Each order is extracted separately and stored in separate extensions in the
MEF file.  The 1D extracted spectrum before telluric correction or flux
calibration, obtained by adding the option
``-p extractSpectra:write_outputs=True`` to the ``reduce`` call.  You can
plot all the orders on a common plot with ``dgsplot``. (The ``--thin`` option
simply plots a thinner line than the default width.)

::

    dgsplot N20260130S0179_H_extracted.fits 1 --thin

.. .. image:: _graphics/gnirsxd_SXD32mm_extracted.png
..   :width: 450
..   :alt: 1D extracted spectrum before telluric correction or flux calibration


The 1D extracted spectrum after telluric correction but before flux
calibration, obtained with ``-p telluricCorrect:write_outputs=True``, looks
like this.

::

    dgsplot N20260130S0179_H_telluricCorrected.fits 1 --thin


.. .. image:: _graphics/gnirsxd_SXD32mm_tellcor.png
..   :width: 600
..   :alt: 1D extracted spectrum after telluric correction or before flux calibration


And the final spectrum, corrected for telluric features and flux calibrated.

::

    dgsplot N20260130S0179_H_1D.fits 1 --thin

.. .. image:: _graphics/gnirsxd_SXD32mm_1d.png
..   :width: 600
..   :alt: 1D extracted spectrum after telluric correction and flux calibration


In the final spectrum, the orders remain separate.  The orders can be combined
into a single spectrum using the ``combineOrders`` primitive.

::

    reduce -r combineOrders N20260130S0179_H_1D.fits
    reduce -r combineOrders N20260130S0179_K_1D.fits


::

    dgsplot N20260130S0179_H_ordersCombined.fits 1 --thin

.. .. image:: _graphics/gnirsxd_SXD32mm_ordersCombined.png
..   :width: 600q
..   :alt: Combined orders. Full range 1D spectrum.
