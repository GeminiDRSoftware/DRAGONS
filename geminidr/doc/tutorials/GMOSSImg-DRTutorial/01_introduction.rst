.. 01_introduction.rst

.. _`DRAGONS`: https://dragons.readthedocs.io/

.. _`GMOS`: https://www.gemini.edu/sciops/instruments/gmos/


.. _introduction:

Introduction
************

This tutorial covers the basics on reducing `GMOS`_ (Gemini Multi-Object
Spectrograph) data using `DRAGONS`_ (Data Reduction for Astronomy from Gemini
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

