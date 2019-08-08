.. 01_introduction.rst

.. _DRAGONS: https://dragons.readthedocs.io/

.. _`Gemini Observatory Archive (GOA)`: https://archive.gemini.edu/

.. _GMOS: https://www.gemini.edu/sciops/instruments/gmos/

.. _introduction:

************
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


.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
package (KL??? name of the package, with URL). Download it and unpack it
somewhere convenient.

.. todo:: add name of and URL to the data package

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvzf KL???

The datasets are found in the subdirectory ``gmosimg_tutorial/playdata``, and we
will work in the subdirectory named ``gmosimg_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
          Archive. Using the tutorial data package is probably more convenient
          but if you really want to learn how to search for and retrieve the
          data yourself, see the step-by-step instructions in the appendix,
          :ref:`goadownload`.


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
