.. datasets.rst

.. _datasetup:

*****************************
Downloading tutorial datasets
*****************************

All the data needed to run this tutorial are found in the tutorial's data
packages.

* Example 1: `f2ls_tutorial_datapkg-JHHK-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/f2ls_tutorial_datapkg-JHHK-v1.tar>`_
* Example 2: `f2ls_tutorial_datapkg-R3KJband-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/f2ls_tutorial_datapkg-R3KJband-v1.tar>`_
* Example 3: `f2ls_tutorial_datapkg-R3KKband-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/f2ls_tutorial_datapkg-R3KKband-v1.tar>`_

Download the package for the example you wish to run and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf gf2ls_tutorial_datapkg-JHHK-v1.tar
    tar xvf f2ls_tutorial_datapkg-R3KJband-v1.tar
    tar xvf f2ls_tutorial_datapkg-R3KKband-v1.tar
    bunzip2 playdata/example*/*.bz2
    rm f2ls_tutorial_datapkg-*.tar

The datasets are found in the subdirectory ``f2ls_tutorial/playdata/example#``, and
we will work in the subdirectory named ``f2ls_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

