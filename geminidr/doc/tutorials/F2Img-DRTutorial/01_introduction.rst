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

This tutorial will use observations from program GS-2013B-Q-15 (PI: Leggett),
NIR photometry of the faint T-dwarf star WISE J041358.14-475039.3, obtained on
2013-Nov-21. Images of this sparse field were obtained in the Y, J, H, Ks bands
using a dither sequence; dayCal DARKS and GCAL flats were obtained as well.
`Leggett, et al. (2015) <https://ui.adsabs.harvard.edu/#abs/2015ApJ...799...37L/abstract>`_
briefly describes the data reduction procedures they followed, which are
similar to those described below.

The first step is to retrieve the data from the `Gemini Observatory Archive
(GOA) <https://archive.gemini.edu/>`_. For more details on using the Archive,
check its `Help Page <https://archive.gemini.edu/help/index.html>`_.

Access the `GOA webpage <https://archive.gemini.edu/>`_, put the data label
**GS-2013B-Q-15-39** in the ``PROGRAM ID`` text field, and press the ``Search``
button in the middle of the page. The page will refresh and display a table with
all the data for this dataset. Since the amount of data is unnecessarily large
for a tutorial (162 files, 0.95 Gb), we will narrow our search by setting the
``Instrument`` drop-down menu to **F2** and the ``Filter`` drop-down menu to
**Y**. Now we have only 9 files, 0.05 Gb.

You can also copy the URL below and paste it on browser to see the search
results:

::

  https://archive.gemini.edu/searchform/GS-2013B-Q-15-39/RAW/cols=CTOWEQ/filter=Y/notengineering/F2/NotFail

At the bottom of the page, you will find a button saying *Download all 9 files
totalling 0.05 Gb*. Click on it to download a `.tar` file with all the data.

The calibration files can be obtained by simply clicking on the *Load Associated
Calibrations* tab, scrolling down to the page and clicking on the *Download all
42 files totalling 0.15 Gb* button.

Finally, you will need a set of short dark frames in order to create the Bad
Pixel Masks (BPM). For that, we will have to perform a search ourselves in the
archive. Fill the search parameters below with their associated values and
click on the ``Search`` button:

- Program ID: GS-CAL20131126-1
- Instrument: F2
- Obs. Type: Dark
- Exposure Time: 3

Here is the associated URL for the search above:

::

  https://archive.gemini.edu/searchform/exposure_time=3/GS-CAL20131126-1/RAW/cols=CTOWEQ/notengineering/F2/NotFail/DARK

Scroll down the page and click on the *Download all 7 files totalling 0.02 Gb*

For convenience, you can also use the three hyperlinks below to download each
tar file.

- `Download Science Data <https://archive.gemini.edu/download/GS-2013B-Q-15-39/filter=Y/RAW/F2/present/NotFail/notengineering/canonical>`_
- `Download Associated Calibrations <https://archive.gemini.edu/download/associated_calibrations/GS-2013B-Q-15-39/filter=Y/RAW/F2/NotFail/notengineering/canonical>`_
- `Download Short DARK data <https://archive.gemini.edu/download/exposure_time=3/GS-CAL20131126-1/RAW/F2/present/NotFail/DARK/notengineering/canonical>`_

Now, copy all the tar files to the same place in your computer. Then use
``tar`` and ``bunzip2`` commands to decompress them:

.. code-block:: bash

    $ cd ${path_to_my_data}/
    $ tar -xf gemini_data.GS-2013B-Q-15-39_F2.tar
    $ tar -xf gemini_calibs.GS-2013B-Q-15-39_F2.tar
    $ tar -xf gemini_data.GS-CAL20131126-1_F2.tar
    $ bunzip2 *.fits.bz2
    $ rm *_flat.fits *_dark.fits  # delete or move reduced data to avoid any confusion

You can add ``-v`` after each command to make it verbose since they can take a
while to be executed. The files names may change depending on the parameters you
used when searching in the `Gemini Archive <https://archive.gemini.edu/searchform>`_.

For this tutorial, we will use a directory to separate the raw data from
the processed data. This is how the data should be organized:

::

  |-- ${path to my data}/
  |   |-- playdata/  # directory for raw data
  |   |-- playground/  # working directory

Use the following commands to have a directory structure consistent the one
used in this tutorial:

.. code-block:: bash

  $ cd ${path_to_my_data}
  $ mkdir playdata  # create directory for raw data
  $ mkdir playground  #  create working directory
  $ mv *.fits ./playdata/  # move all the FITS files to this directory

The full de-compressed data set will have 56 files and use about 0.9 Gb of disk
space.

.. _about_data_set:

About the dataset
=================

The table below contains a summary of the dataset downloaded in the previous
section:

+---------------+---------------------+--------------------------------+
| Science       || S20131121S0075-083 | Y-band, 120 s                  |
+---------------+---------------------+--------------------------------+
| Darks         || S20131127S0257-263 | 3 s, short darks for BPM       |
|               +---------------------+--------------------------------+
|               || S20130930S0242-246 | 20 s, for flat data            |
|               || S20131023S0193-197 |                                |
|               || S20140124S0033-038 |                                |
|               || S20140209S0542-545 |                                |
|               +---------------------+--------------------------------+
|               || S20131120S0115-120 | 120 s, for science data        |
|               || S20131121S0010     |                                |
|               || S20131122S0012     |                                |
|               || S20131122S0438-439 |                                |
+---------------+---------------------+--------------------------------+
| Flats         || S20131129S0320-323 | 20 s, Lamp On, Y-band          |
|               +---------------------+--------------------------------+
|               || S20131126S1111-116 | 20 s, Lamp Off, Y-band         |
+---------------+---------------------+--------------------------------+
