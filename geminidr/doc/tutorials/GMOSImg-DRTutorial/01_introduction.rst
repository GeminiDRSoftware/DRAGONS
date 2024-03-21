.. 01_introduction.rst

.. _`Gemini Observatory Archive (GOA)`: https://archive.gemini.edu/

.. _GMOS: https://www.gemini.edu/sciops/instruments/gmos/

.. _introduction:

************
Introduction
************

This tutorial covers the basics of reducing GMOS_ (Gemini Multi-Object
Spectrographs) data using |DRAGONS|.

Data reduction with DRAGONS can be done in two different ways:

* From the terminal using the command line.
* From Python using the DRAGONS classes and functions.

We show how to run the same reduction using both methods.

* :ref:`starfield_example`
* :ref:`separateCCDs_example`

More examples will be added in the future.

The next two sections explain what are the required software and the data set
that we use throughout the tutorial.

.. _requirements:

Software Requirements
=====================

Before you start, make sure you have |DRAGONS| properly installed and configured
on your machine. You can test that by typing the following commands:

.. code-block:: bash

    $ conda activate dragons
    $ python -c "import astrodata"

Where ``dragons`` is the name of the conda environment where DRAGONS should
be installed. If you have an error message, make sure:

    - Conda is properly installed;

    - A Conda Virtual Environment is properly created and is active;

    - DRAGONS was successfully installed within the Conda Virtual Environment;


.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
packages.  We have split the data packages per example to keep the size
of each package within some reasonable limit.

* Example 1: `gmosim_tutorial_datapkg-starfield-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gmosim_tutorial_datapkg-starfield-v1.tar>`_
* Example 2: `gmosim_tutorial_datapkg-separateCCDs-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gmosim_tutorial_datapkg-separateCCDs-v1.tar>`_

Download one or several packages and unpack them somewhere
convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gmosimg_tutorial_datapkg-starfield-v1.tar
    tar xvf gmosimg_tutorial_datapkg-separateCCD-v1.tar
    bunzip2 gmosimg_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory ``gmosimg_tutorial/playdata/example#``, and we
will work in the subdirectory named ``gmosimg_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
          Archive. Using the tutorial data package is probably more convenient
          but if you really want to learn how to search for and retrieve the
          data yourself, see the step-by-step instructions for Example 1
          in the appendix, :ref:`goadownload`.

