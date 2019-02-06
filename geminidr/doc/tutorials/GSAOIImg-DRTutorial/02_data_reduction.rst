.. 03_data_reduction.rst


.. _command_line_data_reduction:

Data Reduction
**************

This chapter contains a short guide on how to reduce GSAOI data using DRAGONS.

Before start, make sure you have:

    - Conda is properly installed;

    - A Conda Virtual Environment is properly created and is active;

    - AstroConda (STScI) is properly installed within the Virtual Environment;

    - DRAGONS was successfully installed within the Conda Virtual Environment;

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
and will print a table with the file names and the associated tags. Calibration
files will always have the ``CAL`` tag, so we can start getting to know a bit
more about our data set. The output above was trimmed for simplicity.



.. _create_file_lists:

Create File lists
-----------------


.. _process_dark_files:

Process DARK files
------------------


.. _process_bpm_files:

Process BPM files
-----------------


.. _process_flat_files:

Process FLAT files
------------------


.. _processing_science_files:

Process Science files
---------------------

