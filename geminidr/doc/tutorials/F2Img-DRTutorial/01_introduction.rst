.. 01_introduction.rst


.. _introduction:

Introduction
************

This tutorial covers the basics on reducing
`F2 <https://www.gemini.edu/sciops/instruments/flamingos2/>`_ (Gemini South
Adaptive Optics Imager) data using `DRAGONS <https://dragons.readthedocs.io/>`_
(Data Reduction for Astronomy from Gemini Observatory North and South).

The next two sections explain what are the required software and the data set
that we use throughout the tutorial. `Chapter 2: Data Reduction
<command_line_data_reduction>`_ contains a quick example on how to reduce data
using the DRAGONS command line tools. `Chapter 3: Reduction with API
<api_data_reduction>`_ shows how we can reduce the data using DRAGONS' packages
from within Python.


.. _requirements:

Software Requirements
=====================

Before you start, make sure you have `DRAGONS
<https://dragons.readthedocs.io/>`_ properly installed and configured on your
machine. You can test that by typing the following commands:

::

    $ conda activate geminiconda

    $ python -c "import astrodata"

Where ``geminiconda`` is the name of the conda environment where DRAGONS should
be installed. If you have an error message, make sure:

    - Conda is properly installed;

    - A Conda Virtual Environment is properly created and is active;

    - AstroConda (STScI) is properly installed within the Virtual Environment;

    - DRAGONS was successfully installed within the Conda Virtual Environment;


.. _download_sample_files:

Download Sample Files
=====================

This tutorial will use observations from program GS-2013B-Q-15 (PI:Leggett),
NIR photometry of the faint T-dwarf star WISE J041358.14-475039.3, obtained on
2013-Nov-21. Images of this sparse field were obtained in the Y, J, H, Ks bands
using a dither sequence; dayCal DARKS and GCAL flats were obtained as well.
Leggett, et al. (2015; `[L15]
<https://ui.adsabs.harvard.edu/#abs/2015ApJ...799...37L/abstract>`_)
briefly describes the data reduction procedures they followed, which are
similar to those described below.

The first step is to retrieve the data from the Gemini Observatory Archive
(GOA). For more details on using the Archive, check its
`Help Page <https://archive.gemini.edu/help/index.html>`_. The link below takes
you to the result obtained when searching for data that corresponds to the
chosen program.

::

   https://archive.gemini.edu/searchform/GS-2013B-Q-15-39

The bottom of the page contains a button to download the data. You can click on
that, or you can download the images by `clicking directly
here <https://archive.gemini.edu/download/GS-2013B-Q-15-39/present/canonical>`_.
Alternatively, you can download the data by copy-and-pasting the address below
in your browser:

::

   https://archive.gemini.edu/download/GS-2013B-Q-15-39/present/canonical

After retrieving the science data, click the Load Associated Calibrations tab on
the search results page and download the associated dark and flat-field
exposures. Again, the calibration files can be downloaded by `clicking here
<https://archive.gemini.edu/download/associated_calibrations/GS-2013B-Q-15-39/canonical>`_
or by copying the following URL to your browser:

::

    https://archive.gemini.edu/download/associated_calibrations/GS-2013B-Q-15-39/canonical

Unpack all of them in a subdirectory of your working directory (assumed to be
named /raw in this tutorial). Be sure to uncompress the files.

.. code-block:: bash

   $ cd <my_main_working_directory>

   $ tar -xvf *calib*.tar # extract calibration files from .TAR file

   $ tar -xvf *data*.tar # extract science files from .TAR file

   $ bunzip2 *.fits.bz2 # command that will decompress FITS files

   $ mkdir raw/ # create directory named "raw" (optional)

   $ mv *.fits raw/ # move all the raw FITS files to raw (optional)

The full de-compressed data set will have 310 files and use 4.9 Gb of disk
space.
