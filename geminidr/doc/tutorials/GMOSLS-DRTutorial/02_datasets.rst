.. datasets.rst

.. _datasetup:

*****************************
Downloading tutorial datasets
*****************************

All the data needed to run this tutorial are found in the tutorial's data
packages.  We have split the data packages per example to keep the size
of each package within some reasonable limit.

* Example 1: `gmosls_tutorial_datapkg-dithered-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gmosls_tutorial_datapkg-dithered-v1.tar>`_
* Example 2: `gmosls_tutorial_datapkg-largedither-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gmosls_tutorial_datapkg-largedither-v1.tar>`_
* Example 3: `gmosls_tutorial_datapkg-ns-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gmosls_tutorial_datapkg-ns-v1.tar>`_
* Example 4: `gmosls_tutorial_datapkg-nsred-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/gmosls_tutorial_datapkg-nsred-v1.tar>`_

Download one or several packages and unpack them somewhere
convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gmosls_tutorial_datapkg-dithered-v1.tar
    tar xvf gmosls_tutorial_datapkg-ns-v1.tar
    ...
    bunzip2 gmosls_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory ``gmosls_tutorial/playdata/example#``, and
we will work in the subdirectory named ``gmosls_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

