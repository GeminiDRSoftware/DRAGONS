.. 03_data_reduction.rst


.. _command_line_data_reduction:

Data Reduction
**************

DRAGONS installation comes with a set of handful scripts that are used to
reduce astronomical data. One of the most important scripts is called
``reduce``, which is extensively explained in the `Recipe System Users Manual
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/index.html>`_.
For this tutorial, we will be also using other `Supplemental tools
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html>`_,
like ``dataselect``, ``showd``, ``typewalk``, and ``caldb``.

.. todo: write dataselect documentation

.. todo: write showd documentation

.. todo: write typewalk documentation

.. todo: write caldb documentation

http://www.gemini.edu/sciops/data-and-results/processing-software
https://www.gemini.edu/sciops/instruments/gsaoi/calibrations/baseline-calibrations

.. _organize_files:

Organize files
--------------

First of all, let us consider that we have put all the files in the same folder
called ``raw`` and that we do not have any information anymore. From a bash
terminal and from within the Conda Virtual Environment where DRAGONS was
installed, we can call the command tool ``typewalk``:::

    $ typewalk

    directory:  <my_full_path>/raw
     S20150609S0022.fits ............... (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (GEMINI) (GSAOI) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     S20150609S0023.fits ............... (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (GEMINI) (GSAOI) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     S20150609S0024.fits ............... (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (GEMINI) (GSAOI) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     ...
     S20170312S0180.fits ............... (GEMINI) (GSAOI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
     S20170312S0181.fits ............... (GEMINI) (GSAOI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
     S20170312S0198.fits ............... (GEMINI) (GSAOI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
     ...
     S20170315S0286.fits ............... (AZEL_TARGET) (CAL) (DOMEFLAT) (FLAT) (GEMINI) (GSAOI) (IMAGE) (LAMPON) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     S20170316S0090.fits ............... (AZEL_TARGET) (CAL) (DOMEFLAT) (FLAT) (GEMINI) (GSAOI) (IMAGE) (LAMPON) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     S20170316S0091.fits ............... (AZEL_TARGET) (CAL) (DOMEFLAT) (FLAT) (GEMINI) (GSAOI) (IMAGE) (LAMPON) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
     ...

This command will open every FITS file within the current folder (recursively)
and will print a table with the file names and the associated tags. For example,
calibration files will always have the ``CAL`` tag. Flat images will always have
the ``FLAT`` tag. Dark files will have the ``DARK`` tag. This means that we
can start getting to know a bit more about our data set just by looking the
tags. The output above was trimmed for simplicity.


.. _create_file_lists:

Create File lists
-----------------

This data set now contains science and calibration frames. It could have
different observed targets and different exposure times. The current data
reduction pipeline does not organize the data.

That means that we first need to identify these files and create lists that will
be used in the data-reduction process. For that, we will use the ``dataselect``
command line. Please, refer to the `dataselect <>`_ page for details regarding
its usage. Let us start with the DARK files:::

   $ dataselect --tags DARK raw/*.fits > list_of_darks.txt

Here, the ``>`` symbol gets the ``dataselect`` output and stores it within the
``list_of_darks.txt`` file. If you want to see the output, simply omit it and
everything after it.

Now we can do the same with the FLAT files, separating them by filter:::

    $ dataselect --tags FLAT --expr 'filter_name=="Kshort"' raw/*.fits > \
         list_of_Kshort_flats.txt

    $ dataselect --tags FLAT --expr 'filter_name=="H"' raw/*.fits > \
         list_of_H_flats.txt

Recall that the ``\`` (back-slash) is used simply to break the long line .

You can select the standard start with the following command:::

    $ dataselect --expr 'observation_class=="partnerCal"' raw/*.fits
    raw/S20170312S0178.fits
    raw/S20170312S0179.fits
    raw/S20170312S0180.fits
    ...

The problem is that you may have more than one standard star in your data set.
We can verify that by passing the ``dataselect`` output to the ``showd`` command
line using "pipe" (``|``):::

   $ dataselect --expr 'observation_class=="partnerCal"' raw/*.fits | showd -d object

   filename:   object
   ------------------------------
   S20170312S0178.fits: LHS 2026
   S20170312S0179.fits: LHS 2026
   ...
   S20170312S0198.fits: cskd-8
   S20170312S0199.fits: cskd-8
   ...

The ``-d`` flag tells ``showd`` which descriptor will be printed for each input
file. You can create a list for each standard star using the ``object`` descriptor
as an argument for ``dataselect``:::

   $ dataselect --expr 'object=="LHS 2026"' raw/*.fits > list_of_std_LHS_2026.txt

   $ dataselect --expr 'object=="cskd-8"' raw/*.fits > list_of_std_cskd-8.txt

The rest is the data with your science target. Before we create a new list, let
us check if we have more than one target and more than one exposure time:::

   $ dataselect --expr 'observation_class=="science"' raw/*.fits | showd -d object

   filename:   object
   ------------------------------
   S20170505S0095.fits: NGC5128
   S20170505S0096.fits: NGC5128
   ...
   S20170505S0109.fits: NGC5128
   S20170505S0110.fits: NGC5128

We have only one target. Now let us check the exposure time:::

   $ dataselect --expr 'observation_class=="science"' raw/*.fits | showd -d exposure_time

   filename:   exposure_time
   ------------------------------
   S20170505S0095.fits: 60.0
   S20170505S0096.fits: 60.0
   ...
   S20170505S0109.fits: 60.0
   S20170505S0110.fits: 60.0

Again, only one exposure time. Just to show the example, let us consider that
we want to filter all the files whose ``object`` is NGC5128 and that the
``exposure_time`` is 60 seconds. We also want to pass the output to a new list:::

   $ dataselect --expr '(observation_class=="science" and exposure_time==60.)' raw/*.fits > \
      list_of_science_files.txt

.. _process_dark_files:

Process DARK files
------------------

Accordingly to the `Calibration webpage for GSAOI
<https://www.gemini.edu/sciops/instruments/gsaoi/calibrations>`_,
**DARK subtraction is not necessary** since the dark noise level is too low. DARK
files are only used to generate Bad Pixel Masks (BPM).

If, for any reason, you believe that you really need to have a master DARK file,
you can create it using the command below:::

   $ reduce @list_of_darks.txt

Note that ``reduce`` will no separate DARKS with different exposure times. You
will have to create a new list for each exposure time, if that is the case.

Master DARK files can be added to the local database using the ``caldb``
command. Before you run it, make sure you have `configured and initialized your
caldb <>`_. Once you are set, add the Master Dark to the local database using
the following command:::

   $ caldb add ./calibrations/processed_dark/S20150609S0022_dark.fits

Note that the name of the master dark file can be different for you.


.. _create_bpm_files:

Create BPM files
----------------

The Bad Pixel Mask (BPM) files can be created using a set of FLAT images and a
set of DARK files. The FLATs must be obtained in the H band with a number of
counts around 20000 adu and no saturated pixels, usually achieved with 7 seconds
exposure time. The download_sample_files_ contains a sample of the files to be
used in this tutorial. If you need to download files for your own data set, use
the `Gemini Archive Search Form <https://archive.gemini.edu/searchform>`_ to
look for matching data.

The BPM file can be created using the ``makeProcessedBPM`` recipe available
via ``reduce`` command line:::

   $ reduce -r makeProcessedBPM @list_of_H_flats.txt @list_of_darks.txt

The ``-r`` argument tells ``reduce`` which recipe you want to use to replace
the default recipe.


.. _process_flat_files:

Process FLAT files
------------------

FLAT images can be easily reduced using the ``reduce`` command line:::

   $ reduce @list_of_Kshort_flats.txt

If we want ``reduce`` to use the BPM file, we need to add ``-p
addDQ:user_bpm="S20131129S0320_bpm.fits"`` to the command line:::

   $ reduce @list_of_Kshort_flats.txt -p addDQ:user_bpm="S20171208S0053_bpm.fits"

.. note::

   Here we used the "S20171208S0053_bpm.fits" as a BPM file. It is very unlikely
   that your BPM file has the same name. Make sure you use the correct file name.
   Processed BPM files will have the "_bpm.fits" sufix.

Once you finish, you will have the master flat file copied in two places: inside
the same folder where you ran ``reduce`` and inside the
``calibrations/processed_flats/`` folder. Here is an example of a master flat:

.. figure:: _static/img/S20170505S0030_flat.png
   :align: center

   Master Flat - K-Short Band

Note that this figure shows the masked pixels in red color but not all the
detector features are masked. For example, the "Christmas Tree" on the detector
2 can be easily noticed but was not masked.


.. _processing_science_files:

Process Science files
---------------------

Once we have our calibration files processed and added to the database, we can
run ``reduce`` on our science data:::

   $ reduce @list_of_science_files.txt

This command will generate flat corrected and sky subtracted files but will
not stack them. You can find which file is which by its suffix
(``_flatCorrected`` or ``_skySubtracted``).

.. figure:: _static/img/S20170505S0095_skySubtracted.png
   :align: center

   S20170505S0095 - Flat corrected and sky subtracted

The figure above shows an example of a crowded field already reduced. The
masked pixels are represented in white color.

Correct Distortion and Stack Images
-----------------------------------

Finally, you will have to stack your images. For that, you must be aware that
GSAOI images are highly distorted and that this distorion must be corrected
before stacking. At this moment, the standard tool for distortion correction
and image stacking is called `disco-stu`. It can be found
`here <https://www.gemini.edu/sciops/data-and-results/processing-software>`_.
Alternatively, you can copy and paste the URL below into your browser:::

    https://www.gemini.edu/sciops/data-and-results/processing-software

.. todo: Add proper parameter values to ``reduce`` so Sky Subtraction can be
   performed correctly.

.. It's the same as any other IR instrument. It uses the positional offsets to
   work out whether the images all overlap or not. The image with the smallest
   offsets is assumed to contain the science target. If some images are clearly
   in a different position, these are assumed to be sky frames and only these
   are stacked to construct sky frames to be subtracted from the science images.
   If all the images overlap, then all frames can be used to make skies provided
   they're more than a certain distance (a couple of arcseconds) from the
   science frame (to avoid objects falling on top of each other and cancelling
   out).

.. The final reduced data is crap: I have files with no sources or a file
   with a lot of residuum and with a bad WCS. Need to check on this tomorrow.

