.. multisource_cmdline.rst

.. include:: DRAGONSlinks.txt

.. _multisource_cmdline:

********************************************************************
Example 1-A: Multi-source Longslit - Using the "reduce" command line
********************************************************************
In this example we will reduce a GMOS longslit observation of multiple stars
using the "|reduce|" command that is operated directly from the unix shell.
Just open a terminal and load the DRAGONS conda environment to get started.

This observation dithers along the slit and along the dispersion axis.


The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`datamultisource`.

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



Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and a local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows "|reduce|" to make requests for matching **processed**
calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, create or edit the configuration file ``rsys.cfg`` as
follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/gmosls_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

.. note:: ``~`` in the path above refers to your home directory.  Also, don't
    miss the dot in ``.geminidr``.

Then initialize the calibration database::

    caldb init

That's it.  It is ready to use.

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file from the database (it will **not**
remove the file on disk.)  (See the "|caldb|" documentation for more details.)


Create file lists
=================

This data set contains science and calibration frames. For some programs, it
could have different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata.)

First, navigate to the ``playground`` directory in the unpacked data package.


Two lists for the biases
------------------------
We have two sets for biases: one for the science observation, one for the
spectrophotometric standard observation.  The on-sky observations were taken
a month apart, so we will process one master bias of each using contemporary
raw biases.

We will separate the two sets of biases using the UT date.  To inspect the
UT date of the biases so that we can build an appropriate expression for
|dataselect|, we can use the tool |showd| to show descriptor values, in this
case the ``ut_date`` descriptor.  (See the |descriptors| page for a
complete list.)

.. highlight:: text

::

    dataselect ../playdata/*.fits --tags BIAS | showd -d ut_date

    --------------------------------------------
    filename                             ut_date
    --------------------------------------------
    ../playdata/N20180422S0144.fits   2018-04-22
    ../playdata/N20180422S0145.fits   2018-04-22
    ../playdata/N20180422S0146.fits   2018-04-22
    ../playdata/N20180422S0147.fits   2018-04-22
    ../playdata/N20180422S0148.fits   2018-04-22
    ../playdata/N20180423S0148.fits   2018-04-23
    ../playdata/N20180423S0149.fits   2018-04-23
    ../playdata/N20180423S0150.fits   2018-04-23
    ../playdata/N20180423S0151.fits   2018-04-23
    ../playdata/N20180423S0152.fits   2018-04-23
    ../playdata/N20180525S0292.fits   2018-05-25
    ../playdata/N20180525S0293.fits   2018-05-25
    ../playdata/N20180525S0294.fits   2018-05-25
    ../playdata/N20180525S0295.fits   2018-05-25
    ../playdata/N20180525S0296.fits   2018-05-25
    ../playdata/N20180527S0848.fits   2018-05-27
    ../playdata/N20180527S0849.fits   2018-05-27
    ../playdata/N20180527S0850.fits   2018-05-27
    ../playdata/N20180527S0851.fits   2018-05-27
    ../playdata/N20180527S0852.fits   2018-05-27

We can note two groups: one in April 2018, another in May 2018.  We can use
this information to build our two lists of biases.  The April group matches
the standard, the May group matches the science.

::

    dataselect ../playdata/*.fits --tags BIAS --expr='ut_date<="2018-04-30"' -o biasstd.lis
    dataselect ../playdata/*.fits --tags BIAS --expr='ut_date>="2018-05-01"' -o biassci.lis

.. note::  Be mindful of the quotes when using an expression.  The outer quotes
   must be different from the inner quotes.


A list for the flats
--------------------
The GMOS longslit flats are not normally stacked.   The default recipe does
not stack the flats.  This allows us to use only one list of the flats.  Each
will be reduced individually, never interacting with the others.

::

    dataselect ../playdata/*.fits --tags FLAT -o flats.lis


A list for the arcs
-------------------
The GMOS longslit arcs are not normally stacked.  The default recipe does
not stack the arcs.  This allows us to use only one list of arcs.  Each will be
reduce individually, never interacting with the others.

::

    dataselect ../playdata/*.fits --tags ARC -o arcs.lis


A list for the spectrophotometric standard star
-----------------------------------------------
If a spectrophotometric standard is recognized as such by DRAGONS, it will
receive the Astrodata tag ``STANDARD``.  To be recognized, the name of the
star must be in a lookup table.  All spectrophotometric standards normally used
at Gemini are in that table.

::

    dataselect ../playdata/*.fits --tags STANDARD -o std.lis


A list for the science observation
----------------------------------
The science observations are what is left, anything that is not a calibration
or assigned the tag ``CAL``.

First, let's have a look at the list of objects.

::

    dataselect ../playdata/*.fits --xtags CAL | showd -d object

    ---------------------------------------------
    filename                               object
    ---------------------------------------------
    ../playdata/N20180526S1024.fits   1945+4650AB
    ../playdata/N20180526S1025.fits   1945+4650AB
    ../playdata/N20180526S1028.fits   1945+4650AB
    ../playdata/N20180526S1029.fits   1945+4650AB

In this case we only have one target.  If we had more than one, we would need
several lists and we could use the ``object`` descriptor in an expression.  We
will do that here to show how it would be done.  To be clear, the ``--expr``
is not necessary here.

::

    dataselect ../playdata/*.fits --xtags CAL --expr='object=="1945+4650AB"' -o sci.lis


Master Bias
===========
We create the master biases with the "|reduce|" command.  We will run it
twice, once of each of the two raw bias lists, then add the master biases
produced to the local calibration manager with the |caldb| command.

::

    reduce @biasstd.lis
    reduce @biassci.lis
    caldb add *_bias.fits


The two master biases are: ``N20180422S0144_bias.fits`` and
``N20180525S0292_bias.fits``.  The ``@`` character before the name of the input
file is the "at-file" syntax. More details can be found in the |atfile|
documentation.

.. note:: The file name of the output processed bias is the file name of the
    first file in the list with ``_bias`` appended as a suffix.  This the
    general naming scheme used by "|reduce|".


Master Flat Field
=================
GMOS longslit flat field are normally obtained at night along with the
observation sequence to match the telescope and instrument flexure.  The
matching flat nearest in time to the target observation is used to flat field
the target.  The central wavelength, filter, grating, binning, gain, and
read speed must match.

Because of the flexure, GMOS longslit flat field are not stacked.  Each is
reduced and used individually.  The default recipe takes that into account.

We can send all the flats, regardless of characteristics, to |reduce| and each
will be reduce individually.  When a calibration is needed, in this case, a
master bias, the best match will be obtained automatically from the local
calibration manager.

::

    reduce @flats.lis --ql
    caldb add *_flat.fits

We can bulk-add the master flats to the local calibration manager as shown
above.

.. note:: GMOS longslit reduction is currently available only for quicklook
   reduction.  The science quality recipes do not exist, hence the use of the
   ``--ql`` flag to activate the "quicklook" recipes.


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

::

    reduce @arcs.lis --ql
    caldb add *_arcs.fits

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

::

    reduce @std.lis --ql
    caldb add *_standard.fits

We currently do not have tools to inspect the spectra or the calculated
sensitivity function.  In appendix, we show a way to plot them using
matplotlib.

KL?????


Science Observations
====================
The science target is a white dwarfs but there are other stars in the slit too.
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

::

    reduce @sci.lis --ql

This produces a 2-D spectrum (``N20180526S1024_2D.fits``) which has been
biased, flat fielded, QE-corrected, wavelength-calibrated, corrected for
distortion, sky subtracted, and stacked.  It also produces the 1-D spectra
extracted from that 2-D spectrum (``N20180526S1024_1D.fits``).  Each 1-D
spectrum flux calibrated with the sensitivity function from the
spectrophotometric standard. The 1-D spectra are stored as 1-D FITS images in
extensions of the output Multi-Extension FITS file.

This is what the 2-D spectrum looks like.

::

    reduce -r display N20180526S1024_2D.fits


_graphics/???

The aperture found are list in the log for the ``findApertures`` just before
the call to ``traceApertures``.  Information about the apertures are also
available in the header of each extracted spectrum.
And this is what the 1-D flux-calibrated spectrum of the primary target looks
like.

_graphics/???

To learn how to plot a 1-D spectrum with matplotlib, see Appendix ???KL???.

The location