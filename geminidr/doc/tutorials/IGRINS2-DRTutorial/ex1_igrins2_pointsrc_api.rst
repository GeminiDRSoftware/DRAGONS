.. ex1_igrins2_pointsrc_api

.. _igrins2_pointsrc_api:

******************************
Example 1 - Point Source - API
******************************

In this example, we will reduce an IGRINS-2 observation of a CEMP-no star
in a programmatic way with the ``Reduce`` API.

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

Setting up
==========
First navigate to your work directory in the unpacked data package.

::

    cd <path>/igrins2_tutorial/playground

The first steps are to import libraries, set up the calibration manager,
and set the logger.

Configuring the interactive interface
-------------------------------------
In ``~/.dragons/``, add the following to the configuration file ``dragonsrc``::

    [interactive]
    browser = your_preferred_browser

The ``[interactive]`` section defines your preferred browser.  DRAGONS will open
the interactive tools using that browser.  The allowed strings are "**safari**",
"**chrome**", and "**firefox**".

Importing libraries
-------------------

.. code-block:: python
    :linenos:

    import glob

    import astrodata
    import gemini_instruments
    from recipe_system.reduction.coreReduce import Reduce
    from gempy.adlibrary import dataselect

The ``dataselect`` module will be used to create file lists for the
biases, the flats, the arcs, the telluric star, and the science observations.
The ``Reduce`` class is used to set up and run the data
reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger.  (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 7

    from gempy.utils import logutils
    logutils.config(file_name='igrins2_tutorial.log')


Set up the Local Calibration Manager
------------------------------------

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_api`.

We recommend that you clean up your working directory (``playground``) and
create a fresh calibration database (``caldb.init(wipe=True)``) when you
start a new example.


Create file lists
=================

This data set contains science and calibration frames. For some programs, it
could contain different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  However, DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps.
It uses Astrodata tags and |descriptors| to select the files and
send the filenames to a text file that can then be fed to ``Reduce``.  (See the
|astrodatauser| for information about Astrodata and for a list
of |descriptors|.)


Unbundle the Bundles
--------------------
The IGRINS-2 data raw data files are organized in bundles that contain the
data from the H-band arm and the K-band arm.  The first step is to debundle
the data so that the H-band and K-band files are separated.

.. code-block:: python
    :linenos:
    :lineno-start: 9

    all_raw_files = glob.glob('../playdata/example1/*.fits')
    all_raw_files.sort()
    all_bundles = dataselect.select_data(all_raw_files, ['BUNDLE'])

    reduce_debundle = Reduce()
    reduce_debundle.files.extend(all_bundles)
    reduce_debundle.runr()


A list for all H and K channel files
------------------------------------
This list will be used to create the other lists for the flats, sky for
wavelength calibration, telluric star, and science observations.

.. code-block:: python
    :linenos:
    :lineno-start: 16

    all_files = glob.glob('./*.fits')
    all_files.sort()
    all_hk_files = dataselect.select_data(all_files, ['UNPREPARED'])


Two lists for the flats
-----------------------
The IGRINS-2 flats consist of a series of lamp-on and lamp-off exposures.
For this observation, there are 20 lamp-on and 10 lamp-off exposures.  The
software knows how to combine the lamp-on and lamp-off exposures to create
the master flat.  For each arm, we will put all the flats associated with
the science sequence in one list.

.. code-block:: python
    :linenos:
    :lineno-start: 19

    flatsH = dataselect.select_data(
        all_files,
        tags=['FLAT'],
        xtags=[],
        expression=dataselect.expr_parser('band=="H"')
    )
    flatsK = dataselect.select_data(
        all_files,
        tags=['FLAT'],
        xtags=[],
        expression=dataselect.expr_parser('band=="K"')
    )


Two lists for the zenith sky frame
----------------------------------
The zenith sky frame is used to create the wavelength solution using the
sky emission lines instead of the arc lamp lines.  The zenith sky frame is
generally a single exposure.

.. code-block:: python
    :linenos:
    :lineno-start: 31

    skywlenH = dataselect.select_data(
        all_files,
        tags=['SKY'],
        xtags=[],
        expression=dataselect.expr_parser('band=="H"')
    )
    skywlenK = dataselect.select_data(
        all_files,
        tags=['SKY'],
        xtags=[],
        expression=dataselect.expr_parser('band=="K"')
    )


Two lists for the telluric
--------------------------
The IGRINS-2 telluric standards are tagged STANDARD and can be easily separated
from the science observations.

.. code-block:: python
    :linenos:
    :lineno-start: 43

    telluricH = dataselect.select_data(
        all_files,
        tags=['STANDARD'],
        xtags=[],
        expression=dataselect.expr_parser('band=="H"')
    )
    telluricK = dataselect.select_data(
        all_files,
        tags=['STANDARD'],
        xtags=[],
        expression=dataselect.expr_parser('band=="K"')
    )


Two lists for the science observations
--------------------------------------
The science observations are everything not a calibration, ie. not tagged
``CAL``.

.. code-block:: python
    :linenos:
    :lineno-start: 55

    scienceH = dataselect.select_data(
        all_files,
        tags=[],
        xtags=['CAL'],
        expression=dataselect.expr_parser('band=="H"')
    )
    scienceK = dataselect.select_data(
        all_files,
        tags=[],
        xtags=['CAL'],
        expression=dataselect.expr_parser('band=="K"')
    )


Master Flat Field
=================
The IGRINS-2 flats are obtained at the end of the night.  They are a series
of lamp-on and lamp-off exposures.  The software will combine the
lamp-on and lamp-off exposures and create the master flat.

.. code-block:: python
    :linenos:
    :lineno-start: 67

    reduce_flatH = Reduce()
    reduce_flatH.files.extend(flatsH)
    reduce_flatH.runr()

    reduce_flatK = Reduce()
    reduce_flatK.files.extend(flatsK)
    reduce_flatK.runr()

In interactive mode, you can inspect the fit for each
order by selecting the tabs above the plot.

Note that you are not required to run in interactive mode, but you might want
to if flat fielding is critical to your program.

.. code-block:: python
    :linenos:
    :lineno-start: 74

    reduce_flatH = Reduce()
    reduce_flatH.files.extend(flatsH)
    reduce_flatH.uparms = dict([('interactive', True)])
    reduce_flatH.runr()


.. .. image:: _graphics/gnirsxd_interflat.png
..   :width: 600
..   :alt: Interactive flat field inspection.


Wavelength Solution
===================
The wavelength solution is derived from the zenith sky frame.  The sky
emission lines are used to derive the wavelength solution.  The solution is
a 2D fit of all the sky lines in all the orders.

.. note:: The telluric standard will later be used to refine the wavelength
   solution using the telluric absorption.

.. code-block:: python
    :linenos:
    :lineno-start: 78

    reduce_skywlenH = Reduce()
    reduce_skywlenH.files.extend(skywlenH)
    reduce_skywlenH.runr()

    reduce_skywlenK = Reduce()
    reduce_skywlenK.files.extend(skywlenK)
    reduce_skywlenK.runr()


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

.. code-block:: python
    :linenos:
    :lineno-start: 85

    reduce_telluricH = Reduce()
    reduce_telluricH.files.extend(telluricH)
    reduce_telluricH.runr()

    reduce_telluricK = Reduce()
    reduce_telluricK.files.extend(telluricK)
    reduce_telluricK.runr()

You can inspect the telluric fit by activating the interactive mode.

.. code-block:: python
    :linenos:
    :lineno-start: 92

        reduce_telluricH = Reduce()
        reduce_telluricH.files.extend(telluricH)
        reduce_telluricH.uparms = dict([('interactive', True)])
        reduce_telluricH.runr()

The output processed telluric standard will contain the telluric model and the
sensitivity function, as well as a refined wavelength solution derived using
the telluric features.

.. note:: If you wish to override the automatically derived values, or if you
   do not have an internet connection, you can set
   the ``fitTelluric`` primitive parameters yourself.  For example,

   .. code-block:: python

        reduce_telluricH = Reduce()
        reduce_telluricH.files.extend(telluricH)
        reduce_telluricH.uparms = dict([('fitTelluric:bbtemp', 9700),
                                        ('fitTelluric:magnitude', 'H=5.391')])
        reduce_telluricH.runr()

   For K-band, in this case, the magnitude would be set to K=5.333.


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
to build and run the ``Reduce`` object on the science frames to get an
extracted spectrum.

.. code-block:: python
    :linenos:
    :lineno-start: 96

    reduce_scienceH = Reduce()
    reduce_scienceH.files.extend(scienceH)
    reduce_scienceH.runr()

    reduce_scienceK = Reduce()
    reduce_scienceK.files.extend(scienceK)
    reduce_scienceK.runr()

You can also activate the interactive tools to inspect the telluric correction.

.. code-block:: python
    :linenos:
    :lineno-start: 103

    reduce_scienceH = Reduce()
    reduce_scienceH.files.extend(scienceH)
    reduce_scienceH.uparms = dict([('interactive', True)])
    reduce_scienceH.runr()


Each order is extracted separately and stored in separate extensions in the
MEF file.  The 1D extracted spectrum before telluric correction or flux
calibration, obtained by adding the option
``('extractSpectra:write_outputs', True)`` to the ``uparms`` list.

You can plot all the extracted, uncorrected orders on a common plot
with ``dgsplot``.

::

   from gempy.adlibrary import plotting
   ad = astrodata.open('N20260130S0179_H_extracted.fits')
   plotting.dgsplot_matplotlib(ad, 1, kwargs={'linewidth':0.5})

.. .. image:: _graphics/gnirsxd_SXD32mm_extracted.png
..   :width: 450.
..   :alt: 1D extracted spectrum before telluric correction or flux calibration

The 1D extracted spectrum after telluric correction but before flux
calibration, obtained with ``('telluricCorrect:write_outputs', True)``, looks
like this.

::

   from gempy.adlibrary import plotting
   ad = astrodata.open('N20260130S0179_H_telluricCorrected.fits')
   plotting.dgsplot_matplotlib(ad, 1, kwargs={'linewidth':0.5})


.. .. image:: _graphics/gnirsxd_SXD32mm_tellcor.png
..   :width: 600
..   :alt: 1D extracted spectrum after telluric correction or before flux calibration

And the final spectrum, corrected for telluric features and flux calibrated.

::

   from gempy.adlibrary import plotting
   ad = astrodata.open(reduce_scienceH.output_filenames[0])
   plotting.dgsplot_matplotlib(ad, 1, kwargs={'linewidth':0.5})

.. .. image:: _graphics/gnirsxd_SXD32mm_1d.png
..   :width: 600
..   :alt: 1D extracted spectrum after telluric correction and flux calibration

In the final spectrum, the orders remain separate.  The orders can be combined
into a single spectrum using the ``combineOrders`` primitive.

.. code-block:: python
    :linenos:
    :lineno-start: 107

    reduce_combine = Reduce()
    reduce_combine.recipename = 'combineOrders'
    reduce_combine.files.extend([reduce_scienceH.output_filenames[0]])
    reduce_combine.runr()

::

   from gempy.adlibrary import plotting
   ad = astrodata.open('N20260130S0179_H_ordersCombined.fits')
   plotting.dgsplot_matplotlib(ad, 1, kwargs={'linewidth':0.5})

.. .. image:: _graphics/gnirsxd_SXD32mm_ordersCombined.png
..   :width: 600q
..   :alt: Combined orders. Full range 1D spectrum.
