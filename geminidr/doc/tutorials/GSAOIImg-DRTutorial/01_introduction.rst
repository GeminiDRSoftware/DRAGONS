.. 01_introduction.rst


.. _introduction:

Introduction
************

This tutorial covers the basics on reducing
`GSAOI <https://www.gemini.edu/sciops/instruments/gsaoi/>`_ (Gemini South
Adaptive Optics Imager) data using `DRAGONS <https://dragons.readthedocs.io/>`_
(Data Reduction for Astronomy from Gemini Observatory North and South).

The next two sections explain what are the required software and the data set
that we use throughout the tutorial. `Chapter 2: Data Reduction
<command_line_data_reduction>`_ contains a quick example on how to reduce data
using the DRAGONS command line tools.


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

For this tutorial we selected a very sparse field. This makes the sky
subtraction less prone to errors. The selected data set was observed for the
GS-2017A-Q-29 program on the night starting on May 04, 2017.

.. GSAOI images suffer from a lot of distortion. Because of that, we chose to run
   this tutorial on globular clusters that have point sources in the whole
   field-of-view. The selected data set was observed for the GS-2017B-Q-53-15
   program, in Dec 10, 2017, and is related to the published work `Miller, 2019
   <https://ui.adsabs.harvard.edu/#abs/2019AAS...23325007M/abstract>`_.

You can search and download the files on the
`Gemini Archive <https://archive.gemini.edu/searchform>`_ using the
information above. Or simply copy the link below and past to your browser:::

    https://archive.gemini.edu/searchform/cols=CTOWEQ/GS-2017A-Q-29/notengineering/GSAOI/20170504-20170505/science/NotFail

Here is the `link that can be used to download the science files
<https://archive.gemini.edu/download/20170504-20170505/GS-2017A-Q-29/notengineering/GSAOI/science/NotFail/present/canonical>`_
(16 files, 0.37 Gb).

Once you find the data, click on ``Load Associated Calibrations`` to bring up
the associated calibrations. A link to download the data will be in the bottom
of the page. Alternatively, you can simply click on the `link to download the
associated calibration files
<https://archive.gemini.edu/download/associated_calibrations/20170504-20170505/GS-2017A-Q-29/notengineering/GSAOI/science/NotFail/canonical>`_
(34 files, 0.62 Gb).

You might also want to download a set of DARK and H-Band FLAT images in
order to build Bad Pixel Masks (BPM) if you do not have one. The DARK files
can be downloaded using `this link
<https://archive.gemini.edu/download/exposure_time=150/notengineering/GSAOI/Pass/DARK/present/canonical>`_.
or copying and pasting the following address to search form into your browser:::

..    https://archive.gemini.edu/searchform/exposure_time=150/cols=CTOWEQ/notengineering/GSAOI/Pass/DARK


The FLAT files to build the BPM can be downloaded directly by `clicking here
<https://archive.gemini.edu/download/20171201-20171231/object=Domeflat/filter=H/notengineering/GSAOI/Pass/present/canonical>`_
or using the following address:::

..    https://archive.gemini.edu/searchform/object=Domeflat/cols=CTOWEQ/filter=H/notengineering/GSAOI/20171201-20171231/Pass


Copy all the files to the same place in your computer. Then use ``tar`` and
``bunzip2`` to decompress them:::

    $ cd ${path_to_my_data}/
    $ tar -xf gemini_calibs.GS-2017A-Q-29_GSAOI_20170504-20170505.tar
    $ tar -xf gemini_data.GS-2017A-Q-29_GSAOI_20170504-20170505.tar
    $ tar -xf gemini_data.GSAOI.tar
    $ tar -xf gemini_data.GSAOI_20171201-20171231.tar
    $ bunzip2 *.fits.bz2

You can add ``-v`` after each command to make it verbose since they can take a
while to be executed. The files names may change depending on the parameters you
used when searching in the `Gemini Archive <https://archive.gemini.edu/searchform>`_.

For this tutorial, we will store the raw data within the ``./raw/`` directory to
keep things cleaner:::

   $ mkdir ./raw  # create directory
   $ mv *.fits ./raw  # move all the FITS files to this directory
