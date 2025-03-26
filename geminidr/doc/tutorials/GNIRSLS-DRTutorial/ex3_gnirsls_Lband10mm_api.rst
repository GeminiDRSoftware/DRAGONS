.. ex3_gnirsls_Lband10mm_api.rst

.. include:: symbols.txt

.. _gnirsls_Lband10mm_api:

***********************************************************************
Example 3 - L-band Longslit Point Source - Using the "Reduce" class API
***********************************************************************
We will reduce the GNIRS L-band longslit observation of "HD41335", a Be-star,
using the Python
programmatic interface

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
| Telluric            || N20180114S0113-116                          |
+---------------------+----------------------------------------------+
| BPM                 || bpm_20121101_gnirs_gnirsn_11_full_1amp.fits |
+---------------------+----------------------------------------------+

Configuring the interactive interface
=====================================
In ``~/.dragons/``, add the following to the configuration file ``dragonsrc``::

    [interactive]
    browser = your_preferred_browser

The ``[interactive]`` section defines your preferred browser.  DRAGONS will open
the interactive tools using that browser.  The allowed strings are "safari",
"chrome", and "firefox".

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
biases, the flats, the arcs, the standard, and the science observations.
The ``Reduce`` class is used to set up and run the data
reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger.  (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 7

    from gempy.utils import logutils
    logutils.config(file_name='gnirsls_tutorial.log')


Set up the Calibration Service
------------------------------

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_api`.

Create file lists
=================
The next step is to create input file lists.  The module ``dataselect`` helps
with that.  It uses Astrodata tags and |descriptors| to select the files and
store the filenames to a Python list that can then be fed to the ``Reduce``
class. (See the |astrodatauser| for information about Astrodata and for a list
of |descriptors|.)

The first list we create is a list of all the files in the ``playdata``
directory.

.. code-block:: python
    :linenos:
    :lineno-start: 9

    all_files = glob.glob('../playdata/example3/*.fits')
    all_files.sort()

We will search that list for files with specific characteristics.  We use
the ``all_files`` :class:`list` as an input to the function
``dataselect.select_data()`` .  The function's signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')

We show several usage examples below.

A list for the flats
--------------------
The GNRIS flats will be stack together.  Therefore it is important to ensure
that the flats in the list are compatible with each other.  You can use
`dataselect` to narrow down the selection as required.  Here, we have only
the flats that were taken with the science and we do not need extra selection
criteria.

.. code-block:: python
    :linenos:
    :lineno-start: 11

    flats = dataselect.select_data(all_files, ['FLAT'])

A list for the telluric
-----------------------
DRAGONS does not recognize the telluric star as such.  This is because
the observations are taken like science data and the GNIRS headers do not
explicitly state that the observation is a telluric standard.  For now, the
`observation_class` descriptor can be used to differential the telluric
from the science observations, along with the rejection of the `CAL` tag to
reject flats and arcs.

.. code-block:: python
    :linenos:
    :lineno-start: 13

    tellurics = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('observation_class=="partnerCal"')
    )

A list for the science observations
-----------------------------------

In our case, the science observations can be selected from the observation
class, ``science``, that is how they are differentiated from the telluric
standards which are ``partnerCal``.

First, let's have a look at the list of objects.

.. code-block:: python
    :linenos:
    :lineno-start: 19

    all_science = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('observation_class=="science"')
    )
    for sci in all_science:
        ad = astrodata.open(sci)
        print(sci, '  ', ad.object())

::

    ../playdata/example3/N20180114S0121.fits   HD41335
    ../playdata/example3/N20180114S0122.fits   HD41335
    ../playdata/example3/N20180114S0123.fits   HD41335
    ../playdata/example3/N20180114S0124.fits   HD41335

Here we only have one object from the same sequence.  If we had multiple
objects we could add the object name in the expression.

.. code-block:: python
    :linenos:
    :lineno-start: 28

    scitarget = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('object=="HD41335"')
    )

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

.. code-block:: python
    :linenos:
    :lineno-start: 34

    for bpm in dataselect.select_data(all_files, ['BPM']):
        caldb.add_cal(bpm)


Master Flat Field
=================
GNIRS longslit flat field are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.

The GNIRS longslit flatfield requires only lamp-on flats.  Subtracting darks
only increases the noise.

The flats will be stacked.

.. code-block:: python
    :linenos:
    :lineno-start: 36

    reduce_flats = Reduce()
    reduce_flats.files.extend(flats)
    reduce_flats.runr()

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

.. code-block:: python
    :linenos:
    :lineno-start: 39

    reduce_flats = Reduce()
    reduce_flats.files.extend(flats)
    reduce_flats.uparms = dict([('interactive', True)])
    reduce_flats.runr()

.. image:: _graphics/gnirsls_Lband10mm_evenoddflat.png
   :width: 600
   :alt: Even-odd effect in flats


Processed Arc - Wavelength Solution
===================================
The wavelength solution for L-band and M-band data is derived from the peaks
in the telluric transmission in the science frames.  The quality of the
wavelength solution depends on the resolution and strength of the telluric
features.

Wavelength calibration from peaks is better done in interactive mode
despite our efforts to automate the process.

To use the sky transmission peaks in the science frames, we invoke the
``makeWavecalFromSkyEmission`` recipe.

.. code-block:: python
    :linenos:
    :lineno-start: 43

    reduce_sky = Reduce()
    reduce_sky.files.extend(scitarget)
    reduce_sky.recipename = 'makeWavecalFromSkyEmission'
    reduce_sky.uparms = dict([('interactive', True)])
    reduce_sky.runr()

In the L-band, it is very important to inspect the feature identification.
Fortunately, in our case, using the default does lead to a correct feature
identification.

Zooming in:

.. image:: _graphics/gnirsls_Lband10mm_arcID.png
   :width: 600
   :alt: Arc line identifications

.. note:: If the feature identification were to be incorrrect, often changing
    the minimum SNR for peak detection to 5 and recalculating ("Reconstruct points")
    will help find the good solution.


Telluric Standard
=================
The telluric standard observed before the science observation is "hip 28910".
The spectral type of the star is A0V.

To properly calculate and fit a telluric model to the star, we need to know
its effective temperature.  To properly scale the sensitivity function (to
use the star as a spectrophotometric standard), we need to know the star's
magnitude.  Those are inputs to the ``fitTelluric`` primitive.

From Eric Mamajek's list "A Modern Mean Dwarf Stellar Color and Effective
Temperature Sequence"
(https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
we find that the effective temperature of an A0V star is about 9700 K. Using
Simbad, we find that the star has a magnitude of K=4.523.

Note that the data is recognized by Astrodata as normal GNIRS longslit science
spectra.  To calculate the telluric correction, we need to specify the telluric
recipe (``reduceTelluric``), otherwise the default science reduction will be
run.

.. code-block:: python
    :linenos:
    :lineno-start: 48

    reduce_telluric = Reduce()
    reduce_telluric.files.extend(tellurics)
    reduce_telluric.recipename = 'reduceTelluric'
    reduce_telluric.uparms = dict([
                ('fitTelluric:bbtemp', 9700),
                ('fitTelluric:magnitude', 'K=4.523'),
                ('fitTelluric:interactive', True),
                ])
    reduce_telluric.runr()

.. image:: _graphics/gnirsls_Lband10mm_tellfit.png
   :width: 600
   :alt: raw science image

The defaults appear to work well in this case.  The blue end is strongly
affected by the telluric absorption. It is okay for the blue line, the
expected continuum, to be above the data.

Science Observations
====================
The science target is a Be star.  The sequence is one ABBA dither pattern.
DRAGONS will flatfield, wavelength calibrate, subtract the sky, stack the
aligned spectra, extract the source, and finally
remove telluric features and flux calibrate.

This is what one raw image looks like.

.. image:: _graphics/gnirsls_Lband10mm_raw.png
   :width: 400
   :alt: raw science image

With all the calibrations in the local calibration manager, one only needs
to do as follows to reduce the science observations and extract the 1-D
spectrum.

.. code-block:: python
    :linenos:
    :lineno-start: 57

    reduce_science = Reduce()
    reduce_science.files.extend(scitarget)
    reduce_science.runr()

To run the reduction with all the interactive tools activated, set the
``interactive`` parameter to ``True``.

.. code-block:: python
    :linenos:
    :lineno-start: 60

    reduce_science = Reduce()
    reduce_science.files.extend(scitarget)
    reduce_science.uparms = dict([('interactive', True)])
    reduce_science.runr()

The default fits are all good, though the trace can be improved by setting
the order to 5 (interactively or by adding ``('traceApertures:order', 5)``
to the ``uparms`` dictionary).

The 2D spectrum before extraction looks like this, with blue wavelengths at
the bottom and the red-end at the top.

.. image:: _graphics/gnirsls_Lband10mm_2d.png
   :width: 400
   :alt: 2D spectrum

The 1D extracted spectrum before telluric correction or flux calibration,
obtained with ``('extractSpectra:write_outputs', True)``, looks like this.

.. image:: _graphics/gnirsls_Lband10mm_extracted.png
   :width: 600
   :alt: 1D extracted spectrum before telluric correction or flux calibration

The 1D extracted spectrum after telluric correction but before flux
calibration, obtained with ``('telluricCorrect:write_outputs', True)``, looks
like this.

.. image:: _graphics/gnirsls_Lband10mm_tellcor.png
   :width: 600
   :alt: 1D extracted spectrum after telluric correction or before flux calibration

And the final spectrum, corrected for telluric features and flux calibrated.

.. image:: _graphics/gnirsls_Lband10mm_1d.png
   :width: 600
   :alt: 1D extracted spectrum after telluric correction and flux calibration

