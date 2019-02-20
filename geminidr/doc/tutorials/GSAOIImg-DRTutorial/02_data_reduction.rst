.. 03_data_reduction.rst


.. _command_line_data_reduction:

Data Reduction
**************

DRAGONS installation comes with a set of handful scripts that are used to
reduce astronomical data. One of the most important scripts is called
``reduce``, which is extensively explained in the
`Recipe System Users Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/index.html>`_.
For this tutorial, we will be also using other
`Supplemental tools <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html>`_,
like ``dataselect``, ``showd``, ``typewalk``, and ``caldb``.


.. _organize_files:

Organize files
--------------

First of all, let us consider that we have put all the files in the same folder
and that we do not have any information anymore. From a bash terminal and
from within the Conda Virtual Environment where DRAGONS was installed, we can
call the command tool ``typewalk```:::

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
calibration files will always have the ``CAL`` tag. This means that we we can
start getting to know a bit more about our data set just by looking the tags.
The output above was trimmed for simplicity.


.. _create_file_lists:

Create File lists
-----------------

This data set science images obtained with the Kshort and with the J filters. It
also contains images of standard stars obtained in the same night with the same
filters. Finally, it contains flat images in both filters. We first need to
identify these files and create lists that will be used in the data-reduction
process.

Let's first check all the FLAT files:::

    $ dataselect --tags FLAT --expr 'filter_name=="J"' raw/*.fits > flats_J.list

    $ dataselect --tags FLAT --expr 'filter_name=="Kshort"' raw/*.fits > flats_Kshort.list

Everything else is science data:::

    $ dataselect --xtags FLAT raw/*.fits > science.list


.. _process_flat_files:

Process FLAT files
------------------

.. piece of cake. Very easy to do.

.. _processing_science_files:

Process Science files
---------------------

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

