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


Query and Download
==================

This tutorial uses observations from a Science Verification program done during
the commissioning and characterizing phase of the GMOS-N Hamamamatsu CCDs.
The program ID is GN-2017A-SV-151.

The first step of any reduction is to retrieve the data from the
`Gemini Observatory Archive (GOA) <https://archive.gemini.edu/>`_. For more
details on using the Archive, check its
`Help Page <https://archive.gemini.edu/help/index.html>`_.


Science Data
------------

Access the `Gemini Observatory Archive (GOA) <https://archive.gemini.edu/>`_
and fill the search form as follows:

* Program ID: GN-2017A-SV-151-382
* Instrument: GMOS-N
* Filter: i'

Press the ``Search`` button in the middle of the page.

The table will show you 10 files. Mark the checkbox for the first 5 files in
the list.  Normally, you would use all 10 files, but for the purpose of the
tutorial, 5 files will do and will run faster.

You can also copy the URL below and paste it on browser to see the search
results:

..  code-block:: none

    https://archive.gemini.edu/searchform/GN-2017A-SV-151-382/cols=CTOWEQ/filter=i/notengineering/GMOS-N/imaging/science/NotFail


Calibrations
------------

The calibration files could be obtained by simply clicking on the
**Load Associated Calibrations** tab. You will see that the Gemini Archive will
load much more files than we need (129 files, totalling 0.53 Gb). Obviously
we don't need all that.

For this data, we need a few biases and a few twilight flats, all taken around
the time of the observations. How many to download depends on your personal
philosophy to some extend.  For the biases, using 10 to 20 raw biases works
well.  For the twilight flats, make sure that they are set to "Pass", do not
use the "Usable" if you can avoid it.  In this case, because it was
commissioning data, the quality status was not set and all calibrations are
set to "Undefined".  It will be fine for our purpose.

For this tutorial, we will pick the 10 biases taken on the day previous to our
observations since none were taken on the day. The twilight flats from
2017 July 2, GN-CAL20170702-3, are the closest in time to our observations, we
will use those.

For the biases, let's pick the first ten (10) on the list, skipping the very
top one which comes from an engineering program (the GN-ENG- in the program
ID gives it up).  The selected biases are from observation ID GN-CAL20170613-3
and GN-CAL20170615-14.  Select the checkboxes on the left.

For the twilight flats, scroll down the table until you see them, about half
way down.  Be mindful of the last column, we normally must select the
flats with a "Pass" status.  Here all the flats are set to "Undefined" because
this was commissioning data so we will have to make due with them.  Let's pick
the flats from the night of 2017 July 2 with observation ID GN-CAL20170702-3.
Let's pick the first 5 flats.  Select them checkboxes on the left.

Now scroll all the way down and press the "Download Marked Files" button.


Unpacking the data
==================

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
     and create a parallel directory for running the tutorial called
     ``playground``. The tutorial makes assumption as to where everything
     is located.
