.. datasets.rst

.. _datasets:

*****************************
Downloading tutorial datasets
*****************************

.. _datasetup:

Downloading the tutorial datasets
=================================
All the data needed to run this tutorial are found in the tutorial's data
packages.

* Example 1: `niriim_tutorial_datapkg-extended-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/niriim_tutorial_datapkg-extended-v1.tar>`_

Download the package and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf niriim_tutorial_datapkg-extended-v1.tar
    bunzip2 niriimg_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory ``niriimg_tutorial/playdata/example#``, and
we will work in the subdirectory named ``niriimg_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

