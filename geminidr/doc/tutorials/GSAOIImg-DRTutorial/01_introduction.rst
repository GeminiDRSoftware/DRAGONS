.. 01_introduction.rst

.. _introduction:

************
Introduction
************

This tutorial covers the basics of reducing
`GSAOI <https://www.gemini.edu/sciops/instruments/gsaoi/>`_  data using
|DRAGONS|.

The next two sections explain what are the required software and the data set
that we use throughout the tutorial.

The reduction can be done in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

* :ref:`offsetsky_example`

.. _requirements:

Software Requirements
=====================

Before you start, make sure you have |DRAGONS| properly installed and
configured on your machine. You can test that by typing the following commands:

.. code-block:: bash

    $ conda activate dragons
    $ python -c "import astrodata"

Where ``dragons`` is the name of the conda environment where DRAGONS should
be installed. If you have an error message, make sure:

    - Anaconda or MiniConda is properly installed;

    - A Conda Virtual Environment is properly created and is active;

    - DRAGONS was successfully installed within the Conda Virtual Environment;

.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
packages.

* Example 1: `gsaoiim_tutorial_datapkg-offsetsky-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gsaoiim_tutorial_datapkg-offsetsky-v1.tar>`_

Download the package and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gsaoiim_tutorial_datapkg-offsetsky-v1.tar
    bunzip2 gsaoiimg_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory ``gsaoiimg_tutorial/playdata/example#``, and we
will work in the subdirectory named ``gsaoiimg_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
          Archive. Using the tutorial data package is probably more convenient
          but if you really want to learn how to search for and retrieve the
          data yourself, see the step-by-step instructions for Example 1
          in the appendix, :ref:`goadownload`.
