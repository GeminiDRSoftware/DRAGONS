.. 01_introduction.rst

.. _`DRAGONS`: https://dragons.readthedocs.io/

.. _`GMOS`: https://www.gemini.edu/sciops/instruments/gmos/


.. _introduction:

Introduction
************

This tutorial covers the basics on reducing images obtained with the
`GMOS`_ (Gemini Multi-Object Spectrograph) instrument using `DRAGONS`_ (Data
Reduction for Astronomy from Gemini Observatory North and South).

The next two sections explain what are the required software and the data set
that we use throughout the tutorial. `Chapter 2: Data Reduction
<command_line_data_reduction>`_ contains a quick example on how to reduce data
using the DRAGONS command line tools. `Chapter 3: Reduction with API
<api_data_reduction>`_ shows how we can reduce the data using DRAGONS' packages
from within Python.


.. _software_requirements:

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

This tutorial will use data obtained for the GS-2017A-Q-90 program
(P.I.: Walter Weidmann) which consists on broad-band imaging the central star
of some Planetary Nebulaes.

The link for the search form is provided below::

    https://archive.gemini.edu/searchform/RAW/object=PN+G314.5-01.0/cols=CTOWEQ/GS-2017A-Q-90/notengineering/GMOS-S/imaging/science/NotFail/OBJECT

You can copy and paste the URL address on your browser to get all the images,
scroll to the bottom of the page and press the "Download all 36 files totalling
0.20 Gb" button. Then, click on the "Load Associated Calibrations" tab to bring
up a list of calibration files and press the "Download all 418 files totalling
1.64 Gb" button in the bottom of the page to download them.

Alternatively you can use the links below to download the data directly:

    - `Science Data <https://archive.gemini.edu/download/object=PN+G314.5-01.0/notengineering/GS-2017A-Q-90/RAW/GMOS-S/imaging/science/NotFail/OBJECT/present/canonical>`_
    - `Associated Calibration Files <https://archive.gemini.edu/download/associated_calibrations/object=PN+G314.5-01.0/notengineering/GS-2017A-Q-90/RAW/GMOS-S/imaging/science/NotFail/OBJECT/canonical>`_

Copy the two ``.tar`` files to your favorite folder (e.g.: ``~/playground/``).
Decompress them using the following commands::

    $ cd ${path_to_my}/playground
    $ tar -xvf gemini_data.GS-2006B-Q-18_GMOS-S.tar
    $ tar -xvf gemini_calibs.GS-2006B-Q-18_GMOS-S.tar
    $ bzip2 -v *.fits.bz2

In order to keep things organized, let's create a folder called ``raw`` and move
all the raw images into there::

    $ mkdir -p raw/
    $ mv *[0-9].fits ./raw/

If you want to keep the reduced data provided by the archive, you can move them
to a new folder (e.g.: ``red_archive``)::

    $ mkdir -p red_archive/
    $ mv *[0-9]_*.fits red_archive/

Now that we are all set, we can start our data reduction using the DRAGONS
command line tools.