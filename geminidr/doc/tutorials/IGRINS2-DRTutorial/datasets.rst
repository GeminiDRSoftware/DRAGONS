.. datasets.rst

.. _datasetup:

*****************************
Downloading tutorial datasets
*****************************

Data Package
------------
All the data needed to run this tutorial are found in the tutorial's data
package.

* Example 1: `igrins2_tutorial_datapkg-pointsrc-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/igrins2_tutorial_datapkg-pointsrc-v1.tar>`_

Download the package and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf igrins2_tutorial_datapkg-pointsrc-v1.tar
    bunzip2 igrins2_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory
``igrins2_tutorial/playdata/example#``, and we will work in the subdirectory
named ``igrins2_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.
