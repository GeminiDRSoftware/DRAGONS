.. 01_introduction.rst

.. _DRAGONS: https://dragons.readthedocs.io/

.. _`Gemini Observatory Archive (GOA)`: https://archive.gemini.edu/

.. _GMOS: https://www.gemini.edu/sciops/instruments/gmos/

.. _introduction:

Introduction
************

This tutorial covers the basics on reducing GMOS_ (Gemini Multi-Object
Spectrographs) data using DRAGONS_ (Data Reduction for Astronomy from Gemini
Observatory North and South).

The next two sections explain what are the required software and the data set
that we use throughout the tutorial. `Chapter 2: Data Reduction
<command_line_data_reduction>`_ contains a quick example on how to reduce data
using the DRAGONS command line tools. `Chapter 3: Reduction with API
<api_data_reduction>`_ shows how we can reduce the data using DRAGONS' packages
from within Python.


.. _requirements:

Software Requirements
=====================

Before you start, make sure you have DRAGONS_ properly installed and configured
on your machine. You can test that by typing the following commands:

.. code-block:: bash

    $ conda activate geminiconda

.. code-block:: bash

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

..  todo @bquint
..  todo:: @bquint Upload a ``.tar.gz`` file with the full dataset and use the
    URL here.

This tutorial will use observations from program GN-2017A-LP-1 (PI: Wesley
Fraser), "COL-OSSOS: COLours for the Outer Solar System Object Survey", obtained
on 2017-May-25.

Let's start by accessing the `Gemini Observatory Archive (GOA)`_ to download the
data. Then, put the data label **GN-2017A-LP-1-74** in the ``PROGRAM_ID`` text
field, and press the ``Search`` button in the middle of the page. The page will
refresh and display a table with all the data for this dataset.

The table will show you 6 files: five g-band images and one r-band image. We
can exclude the r-band image by selecting **GMOS-N** in the ``Instrument``
drop-down menu. When you do that, the page will display more options. Select
**g'** in the ``Filter`` drop-down that just showed up and press the ``Search``
button again. Now we have only 5 files, 0.10 Gb.

You can also copy the URL below and paste it on browser to see the search
results:

..  code-block:: none

    https://archive.gemini.edu/searchform/GN-2017A-LP-1-74/cols=CTOWEQ/filter=g/notengineering/GMOS-N/NotFail

At the bottom of the page, you will find a button saying ``Download all 5 files
totalling 0.10 Gb`` . Click on it to download a `` .tar `` file with all the
data.

The calibration files could be obtained by simply clicking on the
**Load Associated Calibrations** tab. You will see that the Gemini Archive will
load much more files than we need (239 files, totalling 2.09 Gb). That is too
much for a tutorial so we will look for our calibration files manually.

For the Bias images, fill the search parameters below with their associated
values and click on the ``Search`` button:

- Program ID: GN-CAL20170527-11
- Instrument: GMOS-N
- Binning: 1x1
- Raw / Reduced: Raw Only
- ROI: Full Frame

Once the page reloads, you should see a table with five files. The ``Type``
collumn will tell us that they are all BIAS. Go to the botton of the page and
click on the ``Download all 5 files totalling 0.06 Gb`` .

For the Flat images, fill the search form using the following parameters:

- Program ID: GN-CAL20170530-2
- Instrument: GMOS-N
- Binning: 1x1
- Raw / Reduced: Raw Only
- ROI: Full Frame

Now click on the little black arrow close to the ``Advanced Options`` and change
the ``QA State`` drop-down menu to **Pass** to ensure we have good quality data.

Press the ``Search`` button, the page will reload and show you six
files. The ``Type`` column says **OBJECT** but the ``Object`` columnn says
**Twilight**. This tells us that these are Twilight Flats. Go to the botton of
the page and click on the ``Download alll 6 files totalling 0.20 Gb`` .

For more details on using the Archive, check its
`Help Page <https://archive.gemini.edu/help/index.html>`_.

For convenience, you can also use the three hyperlinks below to download each
tar file.

- `Download Science Data <https://archive.gemini.edu/download/GN-2017A-LP-1-74/filter=g/notengineering/GMOS-N/NotFail/present/canonical>`_
- `Download Bias <https://archive.gemini.edu/download/GN-CAL20170527-11/notengineering/1x1/RAW/GMOS-N/fullframe/NotFail/present/canonical>`_
- `Download Flats <https://archive.gemini.edu/download/GN-CAL20170530-2/notengineering/1x1/RAW/GMOS-N/fullframe/Pass/present/canonical>`_

Now, copy all the ``.tar`` files to the same place in your computer. Then use
``tar`` and ``bunzip2`` commands to decompress them:

.. code-block:: bash

    $ cd ${path_to_my_data}/
    $ tar -xf gemini_data.GN-2017A-LP-1-74_GMOS-N.tar
    $ tar -xf gemini_data.GN-CAL20170527-11_GMOS-N.tar
    $ tar -xf gemini_data.GN-CAL20170530-2_GMOS-N.tar
    $ bunzip2 *.fits.bz2
    $ rm *_flat.fits *_dark.fits  # delete or move reduced data to avoid any confusion

You can add ``-v`` after each command to make it verbose since they can take a
while to be executed. The files names may change depending on the parameters you
used when searching in the `Gemini Archive <https://archive.gemini.edu/searchform>`_.

For this tutorial, we will use a directory to separate the raw data from the
processed data. This is how the data should be organized:

.. code-block:: none

  |-- ${path_to_my_data}/
  |   |-- playdata/  # directory for raw data
  |   |-- playground/  # working directory

Use the following commands to have a directory structure consistent the one
used in this tutorial:

.. code-block:: bash

  $ cd ${path_to_my_data}
  $ mkdir playdata  # create directory for raw data
  $ mkdir playground  #  create working directory
  $ mv *.fits ./playdata/  # move all the FITS files to this directory

The full de-compressed data set will have 16 files and use about 0.7 Gb of disk
space.

.. _about_data_set:

About the dataset
=================

The table below contains a summary of the dataset downloaded in the previous
section:

+---------------+---------------------+--------------------------------+
| Science       || N20170525S0116-120 | 300 s, g-band                  |
+---------------+---------------------+--------------------------------+
| Bias          || N20170527S0528-532 |                                |
+---------------+---------------------+--------------------------------+
| Twilight Flats|| N20170530S0360     | 256 s, g-band                  |
|               || N20170530S0363     | 64 s, g-band                   |
|               || N20170530S0364     | 32 s, g-band                   |
|               || N20170530S0365     | 16 s, g-band                   |
|               || N20170530S0371-372 | 1 s, g-band                    |
+---------------+---------------------+--------------------------------+
