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
and that we do not have any information anymore. From a bash terminal and
from within the Conda Virtual Environment where DRAGONS was installed, we can
call the command tool ``typewalk``:::

    $ typewalk
    ...
    S20171208S0091.fits ............... (AZEL_TARGET) (CAL) (DOMEFLAT) (FLAT) ... (SOUTH) (UNPREPARED)
    S20171208S0092.fits ............... (AZEL_TARGET) (CAL) (DOMEFLAT) (FLAT) ... (SOUTH) (UNPREPARED)
    ...
    S20171210S0042.fits ............... (GEMINI) (GSAOI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
    S20171210S0043.fits ............... (GEMINI) (GSAOI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
    ...

This command will open every FITS file within the current folder (recursively)
and will print a table with the file names and the associated tags. For example,
calibration files will always have the ``CAL`` tag. Flat images will always have
the ``FLAT`` tag. This means that we we can start getting to know a bit more
about our data set just by looking the tags. The output above was trimmed for
simplicity.


.. _create_file_lists:

Create File lists
-----------------

This data set science images obtained with the Kshort and with the J filters. It
also contains images of standard stars obtained in the same night with the same
filters. Finally, it contains flat images in both filters and DARK frames
obtained far in the past. We first need to identify these files and create
lists that will be used in the data-reduction process.

Let us start with the DARK files:::

   $ dataselect --tags DARK raw/*.fits > list_of_dark_files.txt

Now we can do the same with the FLAT files, separating them by filter:::

    $ dataselect --tags FLAT --expr 'filter_name=="J"' raw/*.fits > list_of_J_flats.txt

    $ dataselect --tags FLAT --expr 'filter_name=="Kshort"' raw/*.fits > list_of_Kshort_flats.txt

The rest of the data can be either your science target or other calibration
images, like standard stars. You can select the science files using the following
command:::

    $ dataselect --xtags FLAT raw/*.fits --expr 'observation_class=="science"' \
         > list_of_science_files.txt

Recall that the ``\`` (back-slash) is used simply to break the long line. The
standard stars can be select using the command:::

    $ dataselect --xtags FLAT raw/*.fits \
        --expr 'observation_class=="partnerCal"' > list_of_standard_stars.txt


.. _process_dark_files:

Process DARK files
------------------

Accordingly to the `Calibration webpage for GSAOI
<https://www.gemini.edu/sciops/instruments/gsaoi/calibrations>`_,
DARK subtraction is not necessary since the dark noise level is too low. DARK
files are only used to generate Bad Pixel Masks (BPM). This is described in the
next section.

Create BPM files
----------------



.. _process_flat_files:

Process FLAT files
------------------

FLAT images can be easily reduced using the ``reduce`` command line:::

   $ reduce @list_of_J_flats.lis

   $ reduce @list_of_Kshort_flats.lis

.. _processing_science_files:

Process Science files
---------------------

Once we have our calibration files processed and added to the database, we can
run ``reduce`` on our science data:::

   $ reduce @list_of_science_data

``reduce`` will run on every file within ``list_of_science_data``

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

