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

You might also want to download a set of DARK and H-Band FLAT images in
order to build Bad Pixel Masks (BPM) if you do not have one. The DARK files
can be downloaded using `this link
<https://archive.gemini.edu/download/exposure_time=150/notengineering/GSAOI/Pass/DARK/present/canonical>`_.
or copying and pasting the following address to search form into your browser:::

    https://archive.gemini.edu/searchform/exposure_time=150/cols=CTOWEQ/notengineering/GSAOI/Pass/DARK

The FLAT files to build the BPM can be downloaded directly by
`clicking here
<https://archive.gemini.edu/download/20171201-20171231/object=Domeflat/filter=H/notengineering/GSAOI/Pass/present/canonical>`_
or using the following address:::

    https://archive.gemini.edu/searchform/object=Domeflat/cols=CTOWEQ/filter=H/notengineering/GSAOI/20171201-20171231/Pass

..  Search Form https://archive.gemini.edu/searchform/GS-2017A-Q-44-28/cols=CTOWEQ/notengineering/GSAOI/imaging/20170101-20171201/science/NotFail/OBJECT
    (43 files totalling 0.84 Gb) https://archive.gemini.edu/download/GS-2017A-Q-44-28/20170101-20171201/notengineering/GSAOI/imaging/science/NotFail/OBJECT/present/canonical
    (53 files totalling 0.99 Gb) https://archive.gemini.edu/download/associated_calibrations/GS-2017A-Q-44-28/20170101-20171201/notengineering/GSAOI/imaging/science/NotFail/OBJECT/canonical
