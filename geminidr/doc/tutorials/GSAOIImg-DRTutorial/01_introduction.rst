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

    1. You have Conda properly installed;

    2. You have a conda environment called ``geminiconda`` (or whatever is the
    name of your conda environment);

    3. `AstroConda <https://astroconda.readthedocs.io/>`_ is configured and
    installed within the conda environment.

    4. `DRAGONS <https://dragons.readthedocs.io/>`_ is configured and installed
    within the conda environment.


.. _download_sample_files:

Download Sample Files
=====================

GSAOI images suffer from a lot of distortion. Because of that, we chose to run
this tutorial on globular clusters that have point sources in the whole
field-of-view. The selected data set was observed for the GS-2017B-Q-53-15
program, in Dec 10, 2017, and is related to the published work `Miller, 2019
<https://ui.adsabs.harvard.edu/#abs/2019AAS...23325007M/abstract>`_.

You can search and download the files on the
`Gemini Archive <https://archive.gemini.edu/searchform>`_ using the
information above, or simply copy the link below and past to your browser:::

    https://archive.gemini.edu/searchform/object=NGC+104/cols=CTOWEQ/notengineering/GSAOI/ra=6.0223292/20170201-20171231/science/dec=-72.0814444/NotFail/OBJECT

Here is the `link that can be used to download the science files
<https://archive.gemini.edu/download/sr=180/20170201-20171231/object=NGC+104/notengineering/GSAOI/ra=6.0223292/science/dec=-72.0814444/NotFail/OBJECT/present/canonical>`_
(38 files totalling 0.9 Gb).
And here is the `link to download the associated calibration files
<https://archive.gemini.edu/download/associated_calibrations/sr=180/20170201-20171231/object=NGC+104/notengineering/GSAOI/ra=6.0223292/science/dec=-72.0814444/NotFail/OBJECT/canonical>`_
(61 files totalling 1.02 Gb).
