.. 01_goa_download.rst

.. _goadownload:

***********************************************
Downloading from the Gemini Observatory Archive
***********************************************

For this tutorial we provide a pre-made package with all the necessary data.
Here we show how one can search and download the data directly from the
archive, like one would have to do for their own program.

If you are just interested in trying out the tutorial, we
recommend that you download the pre-made package (:ref:`datasetup`) instead
of getting everything manually.

Step by step instructions
=========================

For this tutorial we selected data observed for for the GS-2017A-Q-29 program on
the night starting on May 04, 2017.

Science data
------------
Access the `GOA webpage <https://archive.gemini.edu/>`_.

In the search form, enter the following information:

* **Program ID**: GS-2017A-Q-29
* **UTC Date**: 20170504-20170505
* **Obs. Class**: science

The search will return 16 files.  Download them all by pressing the
"Download all 16 files" button at the bottom.

Calibrations
------------
Matching calibration files can be obtained by clicking on the *Load Associated
Calibrations* tab. For this data, domeflats (lamp on and off) and a standard
star observation.

The first four files are the standard star sequence.  The other files are
the lamp on and lamp off domeflats.

The table returned by the automatic calibration association has all that we
need.  Download everything by pressing the "Download all 34 files" button at
the bottom.

Unpacking
---------
Now, copy all the ``.tar`` files to the same place in your computer. Then use
``tar`` and ``bunzip2`` commands to decompress them. For example:

.. code-block:: bash

    $ cd ${path_to_my_data}/
    $ tar -xf gemini_data.tar
    $ bunzip2 *.fits.bz2

(The tar files names may differ slightly depending on how you selected and
downloaded the data from the `Gemini Archive <https://archive.gemini.edu/searchform>`_.)

.. note:: If you are using the manually selected data to run the tutorial,
     please remember to put all the data in a directory called ``playdata``,
     and create a parallel directory of running the tutorial called
     ``playground``. The tutorial makes assumption as to where everything
     is located.
