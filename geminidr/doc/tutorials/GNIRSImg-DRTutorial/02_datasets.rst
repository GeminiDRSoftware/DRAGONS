.. datasets.rst

.. _datasets:

********************************
Setting up and tutorial datasets
********************************

.. _datasetup:

Downloading the tutorial datasets
=================================

All the data needed to run this tutorial are found in the tutorial's data
packages.

* Example 1: `gnirsim_tutorial_datapkg-twostars-v2.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gnirsim_tutorial_datapkg-twostars-v2.tar>`_

Download the packages and unpack them somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gnirsim_tutorial_datapkg-twostars-v2.tar
    bunzip2 gnirsimg_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory ``gnirsimg_tutorial/playdata/example#``, and we
will work in the subdirectory named ``gnirsimg_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
   Archive.  Using the tutorial data package is probably more convenient.

