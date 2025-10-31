.. datasets.rst

.. _datasetup:

*****************************
Downloading tutorial datasets
*****************************

All the data needed to run this tutorial are found in the tutorial's data
packages.

* Example 1: `gnirsxd_tutorial_datapkg-SDX32mm-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gnirsxd_tutorial_datapkg-SDX32mm-v1.tar>`_
* Example 2: `gnirsxd_tutorial_datapkg-SDX111mm-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gnirsxd_tutorial_datapkg-SDX111mm-v1.tar>`_

Download the package for the example you wish to run and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gnirsxd_tutorial_datapkg-SDX32mm-v1.tar
    tar xvf gnirsxd_tutorial_datapkg-SDX111mm-v1.tar
    bunzip2 playdata/example*/*.bz2
    rm gnirsxd_tutorial_datapkg-*.tar

The datasets are found in the subdirectory ``gnirsxd_tutorial/playdata/example#``, and
we will work in the subdirectory named ``gnirsxd_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

