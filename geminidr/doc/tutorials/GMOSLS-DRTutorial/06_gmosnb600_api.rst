.. 06_gmosnb600_api.rst

.. include:: DRAGONSlinks.txt

.. _gmosnb600_api:

************************************************************************
Example 2-B: Custom reduction for GMOS-N B600 - Using the "Reduce" class
************************************************************************

A reduction can be initiated from the command line as shown in
:ref:`gmosnb600_cmdline` and it can also be done programmatically as we will
show here.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this example we replicate the command line reduction from
Example 1-A, this time using the Python interface instead of the command line.
Of course what is shown here could be packaged in modules for greater
automation.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`datagmosnb600`.

Here is a copy of the table for quick reference.

+---------------------+---------------------------------+
| Science             || N20180526S1024-1025 (650 nm)   |
|                     || N20180526S1028-1029 (660 nm)   |
+---------------------+---------------------------------+
| Science biases      || N20180525S0292-296             |
|                     || N20180527S0848-852             |
+---------------------+---------------------------------+
| Science flats       || N20180526S1023 (650 nm)        |
|                     || N20180526S1026 (650 nm)        |
|                     || N20180526S1027 (660 nm)        |
|                     || N20180526S1030 (660 nm)        |
+---------------------+---------------------------------+
| Science arcs        || N20180527S0001 (650 nm)        |
|                     || N20180527S0002 (660 nm)        |
+---------------------+---------------------------------+
| Standard (Feige 34) || N20180423S0024 (650 nm)        |
+---------------------+---------------------------------+
| Standard biases     || N20180423S0148-152             |
|                     || N20180422S0144-148             |
+---------------------+---------------------------------+
| Standard flats      || N20180423S0025 (650nm)         |
+---------------------+---------------------------------+
| Standard arc        || N20180423S0110 (650 nm)        |
+---------------------+---------------------------------+


Setting up
==========
First, navigate to your work directory in the unpacked data package.

The first steps are to import libraries, set up the calibration manager,
and set the logger.


Importing libraries
-------------------

.. code-block:: python
    :linenos:

    import glob

    import astrodata
    import gemini_instruments
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system import cal_service
    from gempy.adlibrary import dataselect

The ``dataselect`` module will be used to create file lists for the
darks, the flats and the science observations. The ``cal_service`` package
is our interface to the local calibration database. Finally, the
``Reduce`` class is used to set up and run the data reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger.  (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 8

    from gempy.utils import logutils
    logutils.config(file_name='gmosls_tutorial.log')


Set up the Local Calibration Manager
------------------------------------
DRAGONS comes with a local calibration manager and a local, light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows the ``Reduce`` instance to make requests for matching
**processed** calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, edit the configuration file ``rsys.cfg`` as follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/gmosls_tutorial/playground

This tells the system where to put the calibration database, the
database that will keep track of the processed calibration we are going to
send to it.

.. note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.geminidr``.

The calibration database is initialized and the calibration service is
configured like this:

.. code-block:: python
    :linenos:
    :lineno-start: 10

    caldb = cal_service.CalibrationService()
    caldb.config()
    caldb.init()

    cal_service.set_calservice()

The calibration service is now ready to use.  If you need more details,
check the "|caldb|" documentation in the Recipe System User Manual.


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
    :lineno-start: 16

    all_files = glob.glob('../playdata/*.fits')
    all_files.sort()

We will search that list for files with specific characteristics.  We use
the ``all_files`` :class:`list` as an input to the function
``dataselect.select_data()`` .  The function's signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')

We show several usage examples below.


Two lists for the biases
------------------------
We have two sets for biases: one for the science observation, one for the
spectrophotometric standard observation.  The on-sky observations were taken
a month apart, so we will process one master bias of each using contemporary
raw biases.

We will separate the two sets of biases using the UT date.  To inspect the
UT date of the biases so that we can build an appropriate expression for
``dataselect`` we can print the ``ut_date`` |descriptors|.

.. code-block:: python
    :linenos:
    :lineno-start: 18

    all_biases = dataselect.select_data(all_files, ['BIAS'])
    for bias in all_biases:
        ad = astrodata.open(bias)
        print(bias, '  ', ad.ut_date())

::

    ../playdata/N20180422S0144.fits    2018-04-22
    ../playdata/N20180422S0145.fits    2018-04-22
    ../playdata/N20180422S0146.fits    2018-04-22
    ../playdata/N20180422S0147.fits    2018-04-22
    ../playdata/N20180422S0148.fits    2018-04-22
    ../playdata/N20180423S0148.fits    2018-04-23
    ../playdata/N20180423S0149.fits    2018-04-23
    ../playdata/N20180423S0150.fits    2018-04-23
    ../playdata/N20180423S0151.fits    2018-04-23
    ../playdata/N20180423S0152.fits    2018-04-23
    ../playdata/N20180525S0292.fits    2018-05-25
    ../playdata/N20180525S0293.fits    2018-05-25
    ../playdata/N20180525S0294.fits    2018-05-25
    ../playdata/N20180525S0295.fits    2018-05-25
    ../playdata/N20180525S0296.fits    2018-05-25
    ../playdata/N20180527S0848.fits    2018-05-27
    ../playdata/N20180527S0849.fits    2018-05-27
    ../playdata/N20180527S0850.fits    2018-05-27
    ../playdata/N20180527S0851.fits    2018-05-27
    ../playdata/N20180527S0852.fits    2018-05-27

We can note two groups: one in April 2018, another in May 2018.  We can use
this information to build our two lists of biases.  The April group matches
the standard, the May group matches the science.

.. code-block:: python
    :linenos:
    :lineno-start: 22

    biasstd = dataselect.select_data(
        all_files,
        ['BIAS'],
        [],
        dataselect.expr_parser('ut_date<="2018-04-30"')
    )

    biassci = dataselect.select_data(
        all_files,
        ['BIAS'],
        [],
        dataselect.expr_parser('ut_date>="2018-05-01"')
    )

.. note::  All expression need to be processed with ``dataselect.expr_parser``.


A list for the flats
--------------------
The GMOS longslit flats are not normally stacked.   The default recipe does
not stack the flats.  This allows us to use only one list of the flats.  Each
will be reduced individually, never interacting with the others.

.. code-block:: python
    :linenos:
    :lineno-start: 35

    flats = dataselect.select_data(all_files, ['FLAT'])


A list for the arcs
-------------------
The GMOS longslit arcs are not normally stacked.  The default recipe does
not stack the arcs.  This allows us to use only one list of arcs.  Each will be
reduce individually, never interacting with the others.

.. code-block:: python
    :linenos:
    :lineno-start: 35

    arcs = dataselect.select_data(all_files, ['ARC'])


A list for the spectrophotometric standard star
-----------------------------------------------
If a spectrophotometric standard is recognized as such by DRAGONS, it will
receive the Astrodata tag ``STANDARD``.  To be recognized, the name of the
star must be in a lookup table.  All spectrophotometric standards normally used
at Gemini are in that table.

.. code-block:: python
    :linenos:
    :lineno-start: 36

    stdstar = dataselect.select_data(all_files, ['STANDARD'])

A list for the science observation
----------------------------------
The science observations are what is left, anything that is not a calibration
or assigned the tag ``CAL``.

First, let's have a look at the list of objects.

.. code-block:: python
    :linenos:
    :lineno-start: 37

    all_science = dataselect.select_data(all_files, [], ['CAL'])
    for sci in all_science:
        ad = astrodata.open(sci)
        print(sci, '  ', ad.object())

On line ??, remember that the second argument contains the tags to **include**
(``tags``) and the third argument is the list of tags to **exclude**
(``xtags``).

::

    ../playdata/N20180526S1024.fits    1945+4650AB
    ../playdata/N20180526S1025.fits    1945+4650AB
    ../playdata/N20180526S1028.fits    1945+4650AB
    ../playdata/N20180526S1029.fits    1945+4650AB

In this case we only have one target.  If we had more than one, we would need
several lists and we could use the ``object`` descriptor in an expression.  We
will do that here to show how it would be done.  To be clear, the
``dataselect.expr_parser`` argument is not necessary in this specific case.

.. code-block:: python
    :linenos:
    :lineno-start: 42

    scitarget = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('object=="1945+4650AB"')
    )


Master Bias
===========
We create the master biases with the ``Reduce`` class.  We will run it
twice, once of each of the two raw bias lists, then add the master biases
produced to the local calibration manager with the ``caldb`` instance.
The output is written to disk and its name is
stored in the ``Reduce`` instance.  The calibration service expects the
name of a file on disk.

.. code-block:: python
    :linenos:
    :lineno-start: 48

    reduce_biasstd = Reduce()
    reduce_biassci = Reduce()
    reduce_biasstd.files.extend(biasstd)
    reduce_biassci.files.extend(biassci)
    reduce_biasstd.runr()
    reduce_biassci.runr()

    caldb.add_cal(reduce_biasstd.output_filenames[0])
    caldb.add_cal(reduce_biassci.output_filenames[0])

The two master biases are: ``N20180422S0144_bias.fits`` and
``N20180525S0292_bias.fits``.


.. note:: The file name of the output processed bias is the file name of the
    first file in the list with ``_bias`` appended as a suffix.  This the
    general naming scheme used by the ``Recipe System``.

Master Flat Field
=================
GMOS longslit flat field are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.  The
matching flat nearest in time to the target observation is used to flat field
the target.  The central wavelength, filter, grating, binning, gain, and
read speed must match.

Because of the flexure, GMOS longslit flat field are not stacked.  Each is
reduced and used individually.  The default recipe takes that into account.

We can send all the flats, regardless of characteristics, to ``Reduce`` and each
will be reduce individually.  When a calibration is needed, in this case, a
master bias, the best match will be obtained automatically from the local
calibration manager.

.. code-block:: python
    :linenos:
    :lineno-start: 52

    reduce_flats = Reduce()
    reduce_flats.files.extend(flats)
    reduce_flats.mode = 'ql'
    reduce_flats.runr()

    for f in reduce_flats.output_filenames:
        caldb.add_cal(f)


.. note:: GMOS longslit reduction is currently available only for quicklook
   reduction.  The science quality recipes do not exist, hence the use of the
   ``ql`` mode to activate the "quicklook" recipes.


Processed Arc - Wavelength Solution
===================================
GMOS longslit arc can be obtained at night with the observation sequence,
if requested by the program, but are often obtained at the end of the night
instead.  Like the spectroscopic flats, they are not stacked which means that
they can be sent to reduce all to together and will be reduced individually.

The wavelength solution is automatically calculated and has been found to be
quite reliable.  There might be cases where it fails; inspect the
``*_mosaic.pdf`` plot and the RMS of ``determineWavelengthSolution`` in the
logs to confirm a good solution.

.. code-block:: python
    :linenos:
    :lineno-start: 56

    reduce_arcs = Reduce()
    reduce_arcs.files.extend(arcs)
    reduce_arcs.mode = 'ql'
    reduce_arcs.runr()

    for f in reduce_arcs.output_filenames:
        caldb.add_cal(f)

.. note:: Failures of the wavelength solution calculation are not easy to fix
   in quicklook mode.  It might be better to simply not use the arc at all and
   rely on the approximate solution instead.  When the science quality package
   is released, there will be interactive tools to fix a bad solution.
   Remember, this is version only offers quicklook reduction for GMOS longslit.

Processed Standard - Sensitivity Function
=========================================
The GMOS longslit spectrophotometric standards are normally taken when there
is a hole in the queue schedule, often when the weather is not good enough
for science observations.  One standard per configuration, per program is
the norm.  If you dither along the dispersion axis, mostly likely only one
of the positions will have been used for the spectrophotometric standard.
This is normal for baseline calibrations at Gemini.  The standard is used
to calculate the sensitiviy function.  It has been shown that a difference of
10 or so nanometer does not significantly impact the spectrophotometric
calibration.

The reduction of the standard will be using a master bias, a master flat,
and a processed arc.  If those have been added to the local calibration
manager, they will be picked up automatically.

.. code-block:: python
    :linenos:
    :lineno-start: 60

    reduce_std = Reduce()
    reduce_std.files.extend(stdstar)
    reduce_std.mode = 'ql'
    reduce_std.runr()

    caldb.add_cal(reduce_std.output_filenames[0])

We currently do not have tools to inspect the spectra or the calculated
sensitivity function.  In the Tips and Tricks chapter, we show a way to plot them using
matplotlib.

KL?????


Science Observations
====================
The science target is a well-separated binary white dwarfs but there are
other stars in the slit too.
DRAGONS will extract everything it can find an aperture for.  The sequence
has four images that were dithered spatially and along the dispersion axis.
DRAGONS will register the four images in both direction, align and stack them
before extracting the 1-D spectra.

This is what one raw image looks like.

_graphics/rawscience.png

With the master bias, the master flat, the processed arcs (one for each of the
grating position, aka central wavelength), and the processed standard in the
local calibration manager, to reduce the science observations and extract 1-D
spectra, one only needs to do as follow.

.. code-block:: python
    :linenos:
    :lineno-start: 64

    reduce_science = Reduce()
    reduce_science.files.extend(scitarget)
    reduce_science.mode = 'ql'
    reduce_science.runr()

This produces a 2-D spectrum (``N20180526S1024_2D.fits``) which has been
biased, flat fielded, QE-corrected, wavelength-calibrated, corrected for
distortion, sky subtracted, and stacked.  It also produces the 1-D spectra
extracted from that 2-D spectrum (``N20180526S1024_1D.fits``).  Each 1-D
spectrum flux calibrated with the sensitivity function from the
spectrophotometric standard. The 1-D spectra are stored as 1-D FITS images in
extensions of the output Multi-Extension FITS file.

This is what the 2-D spectrum looks like.

.. code-block:: python
    :linenos:
    :lineno-start: 70

    display = Reduce()
    display.files = ['N20180526S1024_2D.fits']
    display.recipename = 'display'
    display.runr()

_graphics/???

The aperture found are list in the log for the ``findApertures`` just before
the call to ``traceApertures``.  Information about the apertures are also
available in the header of each extracted spectrum.
And this is what the 1-D flux-calibrated spectrum of the primary target looks
like.

_graphics/???

To learn how to plot a 1-D spectrum with matplotlib, see Tips and Tricks ???KL???.

