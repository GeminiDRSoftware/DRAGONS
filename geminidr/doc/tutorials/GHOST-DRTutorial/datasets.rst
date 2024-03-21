.. datasets.rst

.. _datasetup:

*****************************
Downloading tutorial datasets
*****************************

Data Package
------------
All the data needed to run this tutorial are found in the tutorial's data
package.

* Example 1: `ghost_tutorial_datapkg-stdonetarget-v1.tar <https://www.gemini.edu/sciops/data/software/datapkgs/ghost_tutorial_datapkg-stdonetarget-v1.tar>`_

Download the package and unpack it somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvf ghost_tutorial_datapkg-stdonetarget-v1.tar
    bunzip2 ghost_tutorial/playdata/example*/*.bz2

The datasets are found in the subdirectory ``ghost_tutorial/playdata/example#``, and
we will work in the subdirectory named ``ghost_tutorial/playground``.

.. note:: All the raw data can also be downloaded from the Gemini Observatory
     Archive.  Using the tutorial data package is probably more convenient.

Bad Pixel Masks
---------------
The BPMs are not included in the data package.

.. If you add the
  ``archive.gemini.edu`` in the list of databases in ``.dragonsrc``, they will be
  automatically associated and retrieved.  See :ref:`cal_service_config`.

You can download them from
here: https://archive.gemini.edu/searchform/GHOST/notengineering/cols=CTOWBEQ/NotFail/not_site_monitoring/BPM

Put them in ``playdata/example1/``.
