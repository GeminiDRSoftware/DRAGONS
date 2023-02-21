.. datasets.rst

.. _datasetup:

*****************************
Downloading tutorial datasets
*****************************

.. todo:: Consider one data pkg per example.  Everything in one might get big.

All the data needed to run this tutorial are found in the tutorial's data
package:

    `<http://www.gemini.edu/sciops/data/software/datapkgs/gmosls_tutorial_datapkg-v1.tar>`_

Download it and unpack it somewhere convenient.

.. highlight:: bash

.. todo:: UPDATE datapkg to include N&S data. UPDATE version number
.. todo:: Add the BPMs to the datapkg.

::

    cd <somewhere convenient>
    tar xvf gmosls_tutorial_datapkg-v1.tar
    bunzip2 gmosls_tutorial/playdata/*.bz2

The datasets are found in the subdirectory ``gmosls_tutorial/playdata``, and
we will work in the subdirectory named ``gmosls_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

