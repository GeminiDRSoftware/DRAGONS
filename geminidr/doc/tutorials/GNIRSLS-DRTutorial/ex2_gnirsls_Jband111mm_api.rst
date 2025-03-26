.. ex2_gnirsls_Jband111mm_api.rst

.. include:: symbols.txt

.. _gnirsls_Jband111mm_api:

***********************************************************************
Example 2 - J-band Longslit Point Source - Using the "Reduce" class API
***********************************************************************
In this example, we will reduce the GNIRS J-band longslit observation of
a metal-poor M dwarf, using the Python
programmatic interface

This observation uses the 111 l/mm grating, the short-blue camera, a 0.3 arcsec
slit, and is set to a central wavelength of 1.22 |um|.   The dither pattern is
two consecutive ABBA sequences.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`gnirsls_Jband111mm_dataset`

Here is a copy of the table for quick reference.

+---------------------+----------------------------------------------+
| Science             || N20180201S0052-59                           |
+---------------------+----------------------------------------------+
| Science flats       || N20180201S0060-64                           |
+---------------------+----------------------------------------------+
| Science arcs        || N20180201S0065                              |
+---------------------+----------------------------------------------+
| Telluric            || N20180201S0071-74                           |
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

    all_files = glob.glob('../playdata/example2/*.fits')
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

A list for the arcs
-------------------
The GNIRS longslit arc was obtained at the end of the science observation.
Often two are taken.  We will use both in this case and stack them later.

.. code-block:: python
    :linenos:
    :lineno-start: 12

    arcs = dataselect.select_data(all_files, ['ARC'])


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

The science observations can be selected from the observation
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

    ../playdata/example2/N20180201S0052.fits   target_37
    ../playdata/example2/N20180201S0053.fits   target_37
    ../playdata/example2/N20180201S0054.fits   target_37
    ../playdata/example2/N20180201S0055.fits   target_37
    ../playdata/example2/N20180201S0056.fits   target_37
    ../playdata/example2/N20180201S0057.fits   target_37
    ../playdata/example2/N20180201S0058.fits   target_37
    ../playdata/example2/N20180201S0059.fits   target_37

Here we only have one object from the same sequence.  If we had multiple
objects we could add the object name in the expression.

.. code-block:: python
    :linenos:
    :lineno-start: 28

    scitarget = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('object=="target_37"')
    )

Bad Pixel Mask
==============
Starting with DRAGONS v3.1, the static bad pixel masks (BPMs) are now handled
as calibrations.  They
are downloadable from the archive instead of being packaged with the software.
They are automatically associated like any other calibrations.  This means that
the user now must download the BPMs along with the other calibrations and add
the BPMs to the local calibration manager.

See :ref:`getBPM` in :ref:`tips_and_tricks` to learn about the various ways
to get the BPMs from the archive.

To add the BPM included in the data package to the local calibration database:

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
two sets of points, with a symmetrical residual fit.

Note that you are not required to run in interactive mode, but you might want
to if flat fielding is critical to your program.

.. code-block:: python
    :linenos:
    :lineno-start: 39

    reduce_flats = Reduce()
    reduce_flats.files.extend(flats)
    reduce_flats.uparms = dict([('interactive', True)])
    reduce_flats.runr()

In this case, order=20, the default, worked well and we find that sigma
clipping with 1 iteration and grow=2 rejects the outliers at the left end
of the flat.  The fit leads to residuals that are symmetrical.

.. image:: _graphics/gnirsls_Jband111_evenoddflat.png
   :width: 600
   :alt: Even-odd effect in flats


.. _gnirsls_Jband111mm_api_arc:

Processed Arc - Wavelength Solution
====================================
Obtaining the wavelength solution for GNIRS longslit data can be a complicated
topic.  The quality of the results and what to use depend greatly on the
wavelength regime and the grating.

Our configuration in this example is J-band with a central wavelength of
1.22 |um|, using the 111 l/mm grating. Arcs are available, however, depending
on the central wavelength setting, there might be cases where there are too
few lines or the coverage is not adequate to get a good solution.

In our current case, the numbers of arcs lines and sky lines are similar.
Either solution could work.  We will show the result of both.  It is up to
the user to decide which solution is best for their science.

Using the arc lamp
------------------
Because the slit length does not cover the whole array, we want to know where
the unilluminated areas are located and ignore them when the distortion
correction is calculated (along with the wavelength solution).  That information
is measured during the creation of the flat field and stored in the processed
flat.   Right now, the association rules do not automatically associate
flats to arcs, therefore we need to specify the processed flat on the
command line.  Using the flat is optional but it is recommended when using
an arc lamp.

.. code-block:: python
    :linenos:
    :lineno-start: 43

    reduce_arcs = Reduce()
    reduce_arcs.files.extend(arcs)
    reduce_arcs.uparms = dict([
                ('flatCorrect:flat', reduce_flats.output_filenames[0]),
                ('interactive', True),
                ])
    reduce_arcs.runr()

Here, increasing the order to 4 helps to get a tighter fit.

.. image:: _graphics/gnirsls_Jband111mm_arcID.png
    :width: 600
    :alt: Arc line identifications

.. image:: _graphics/gnirsls_Jband111mm_arcfit.png
   :width: 600
   :alt: Arc line fit

Using the sky lines
-------------------
The spectrum has a number of OH and O\ :sub:`2`\  sky lines that can be used to
create a wavelength solution.  The calibration can be done on a single frame
or, in case of multiple input frames, the frames will be stacked.  It is
recommended to use only one frame for a more precise wavelength solution,
unless multiple frames are needed to increase the signal-to-noise ratio.  Here
we will use all the frames in the ``sci.lis`` list.

Wavelength calibration from sky lines is better done in interactive mode
despite our efforts to automate the process.

To use the sky lines in the science frames instead of the lamp arcs, we
invoke the ``makeWavecalFromSkyEmission`` recipe.

.. code-block:: python
    :linenos:
    :lineno-start: 50

    reduce_sky = Reduce()
    reduce_sky.files.extend(scitarget)
    reduce_sky.recipename = 'makeWavecalFromSkyEmission'
    reduce_sky.uparms = dict([('interactive', True)])
    reduce_sky.runr()

In this case, using all the frames, we get a good signal to noise and an
automatic fit.  If you wanted, you could identify more sky lines manually.

.. image:: _graphics/gnirsls_Jband111_skylineref.png
   :width: 600
   :alt: Sky lines identification with reference spectrum

.. image:: _graphics/gnirsls_Jband111_skyfit.png
   :width: 600
   :alt: Sky lines fit

.. note::  If the sky lines were too weak and no fit were found, a possible
    solution is to lower the minimum SNR to 5 (down from the default of 10).
    This setting is in the left control panel.  When done, click the the
    "Reconstruct points" button.

    When lowering the SNR, lowering the
    high and low sigma clipping to 2 will help reject some of the weak
    blended lines that are more inaccurate.

Which solution to use?
----------------------
Each case will be slightly different.   Whether you decide to use the solution
from the arc lamp or the sky lines is up to you.

Once you have decided, we recommend that you remove the one you do not want
to use from the calibration manager database.  Since the ``_arc`` file selected
will always be the "closest in time" to the science observation, there might be
cases where the lamp solution will be picked for the last datasets in the
sequence while the sky lines solution will be picked for the first datasets in
the sequence.

So pick one, remove the other.

.. code-block:: python
    :linenos:
    :lineno-start: 55

    caldb.remove_cal(reduce_arcs.output_filenames[0])
    ... or ...
    caldb.remove_cal(reduce_sky.output_filenames[0])

In this tutorial, we remove the lamp solution.

If you were to want to try it anyway for telluric standard reduction
or the science reduction, you can force its use by setting the
``ucals`` attribute, eg.
``reduce_telluric.ucals = dict([('processed_arc', reduce_arcs.output_filenames[0])])``

Telluric Standard
=================
The telluric standard observed before the science observation is "hip 55627".
The spectral type of the star is A0V.

To properly calculate and fit a telluric model to the star, we need to know
its effective temperature.  To properly scale the sensitivity function (to
use the star as a spectrophotometric standard), we need to know the star's
magnitude.  Those are inputs to the ``fitTelluric`` primitive.

From Eric Mamajek's list "A Modern Mean Dwarf Stellar Color and Effective
Temperature Sequence"
(https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
we find that the effective temperature of an A0V star is about 9700 K. Using
Simbad, we find that the star has a magnitude of K=9.165.

Note that the data is recognized by Astrodata as normal GNIRS longslit science
spectra.  To calculate the telluric correction, we need to specify the telluric
recipe (``reduceTelluric``), otherwise the default science reduction will be
run.

.. code-block:: python
    :linenos:
    :lineno-start: 58

    reduce_telluric = Reduce()
    reduce_telluric.files.extend(tellurics)
    reduce_telluric.recipename = 'reduceTelluric'
    reduce_telluric.uparms = dict([
                ('fitTelluric:bbtemp', 9700),
                ('fitTelluric:magnitude', 'K=9.165'),
                ('fitTelluric:interactive', True),
                ])
    reduce_telluric.runr()

Adjusting the order of the spline to 8 leads to more randomized residuals
(second panel).

.. image:: _graphics/gnirsls_Jband111mm_tellfit.png
   :width: 600
   :alt: raw science image

Science Observations
====================
The science target is a low metallicity M-dwarf.  The sequence is two ABBA
dithered observations.  DRAGONS will flat field, wavelength calibrate,
subtract the sky, stack the aligned spectra, extract the source, and finally
remove telluric features and flux calibrate.

Following the wavelength calibration, the default recipe has an optional
step to adjust the wavelength zero point using the sky lines.  By default,
this step will NOT make any adjustment.  We found that in general, the
adjustment is so small as being in the noise.  If you wish to make an
adjustment, or try it out, see :ref:`wavzero` to learn how.

This is what one raw image looks like.

.. image:: _graphics/gnirsls_Jband111mm_raw.png
   :width: 400
   :alt: raw science image


With all the calibrations in the local calibration manager, one only needs
to do as follows to reduce the science observations and extract the 1-D
spectrum.

.. code-block:: python
    :linenos:
    :lineno-start: 67

    reduce_science = Reduce()
    reduce_science.files.extend(scitarget)
    reduce_science.runr()

.. todos:: Crashes.  Is that due to the apertures?  Work when
    I correct the apertures interatively.  Can it deal with more
    than one aperture?

To run the reduction with all the interactive tools activated, set the
``interactive`` parameter to ``True``.

.. code-block:: python
    :linenos:
    :lineno-start: 70

    reduce_science = Reduce()
    reduce_science.files.extend(scitarget)
    reduce_science.uparms = dict([('interactive', True)])
    reduce_science.runr()

The 2D spectrum, without telluric correction and flux calibration, looks
like this:

.. code-block:: python
    :linenos:
    :lineno-start: 74

    reduce_display = Reduce()
    reduce_display.files = ['N20180201S0052_2D.fits']
    reduce_display.recipename = 'display'
    reduce_display.runr()

.. image:: _graphics/gnirsls_Jband111mm_2D.png
   :width: 400
   :alt: reduced 2D spectrum

The 1D extracted spectrum before telluric correction or flux calibration,
obtained by adding ``('extractSpectra:write_outputs', True)`` to the
``uparms`` dictionary, looks like this.

.. image:: _graphics/gnirsls_Jband111mm_extracted.png
   :width: 600
   :alt: 1D extracted spectrum before telluric correction or flux calibration

The 1D extracted spectrum after telluric correction but before flux
calibration, obtained by adding ``('telluricCorrect:write_outputs', True)`` to the
``uparms`` dictionary, looks
like this.

.. image:: _graphics/gnirsls_Jband111mm_tellcor.png
   :width: 600
   :alt: 1D extracted spectrum after telluric correction or before flux calibration

And the final spectrum, corrected for telluric features and flux calibrated.

.. image:: _graphics/gnirsls_Jband111mm_1d.png
   :width: 600
   :alt: 1D extracted spectrum after telluric correction and flux calibration
