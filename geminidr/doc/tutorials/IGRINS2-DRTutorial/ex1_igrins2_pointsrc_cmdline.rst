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


Bundles
-------
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

    reduce @flatH.lis
    reduce @flatK.lis

While the non-interactive solution should be correct, if you wish to inspect
the fit run the flat normalization in interactive mode by add
`-p interactive=True` to the command.

::

    reduce @flatH.lis -p interactive=True

.. .. image:: _graphics/igrins2_interflat.png
..   :width: 600
..   :alt: Interactive flat field inspection.

.. todo:: Issues with the interactive tool and the fit.  This is using
   branch tutorial/from_feature_igrins2.  The tabs for each order are not
   working properly.  There's one unreadable tab and a scrollbar.  As for the
   fit, it is too low, it is being dragged down bye the absorption lines.
   What are those absorption lines anyway?

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

To inspect the wavelength solution, run the command in interactive mode by
adding `-p interactive=True` to the command.

::

    reduce @skywlenH.lis -p interactive=True
