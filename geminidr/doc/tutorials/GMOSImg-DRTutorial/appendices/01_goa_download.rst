.. goadownload.rst

.. _goadownload:

***********************************************
Downloading from the Gemini Observatory Archive
***********************************************

For this tutorial we provide a pre-made package with all the necessary data.
Here we show how one can search and download the data directly from the
archive, like one would have to do for their own program.

If you are just interested in trying out the tutorial, we still
recommended that you download the pre-made package (:ref:`datasetup`) instead
of getting everything manually.


Query and Download
==================

This tutorial will use observations from program GN-2017A-LP-1 (PI: Wesley
Fraser), "COL-OSSOS: COLours for the Outer Solar System Object Survey", obtained
on 2017-May-25.

The first step of any reduction is to retrieve the data from the
`Gemini Observatory Archive (GOA) <https://archive.gemini.edu/>`_. For more
details on using the Archive, check its
`Help Page <https://archive.gemini.edu/help/index.html>`_.


Science Data
------------

Once you are in the `Gemini Observatory Archive (GOA) <https://archive.gemini.edu/>`_
put the data label **GN-2017A-LP-1-74** in the ``PROGRAM_ID`` text field, and
press the ``Search`` button in the middle of the page. The page will refresh and
display a table with all the data for this dataset.

The table will show you 6 files: five g-band images and one r-band image. We
can exclude the r-band image by selecting **GMOS-N** in the ``Instrument``
drop-down menu. When you do that, the page will display more options. Select
**g'** in the ``Filter`` drop-down that just showed up and press the ``Search``
button again. Now we have only 5 files, 0.10 Gb.

You can also copy the URL below and paste it on browser to see the search
results:

..  code-block:: none

    https://archive.gemini.edu/searchform/GN-2017A-LP-1-74/cols=CTOWEQ/filter=g/notengineering/GMOS-N/NotFail

At the bottom of the page, you will find a button saying ``Download all 5 files
totalling 0.10 Gb`` . Click on it to download a `` .tar `` file with all the
data.


Calibrations
------------

The calibration files could be obtained by simply clicking on the
**Load Associated Calibrations** tab. You will see that the Gemini Archive will
load much more files than we need (239 files, totalling 2.09 Gb). That is too
much for a tutorial so we will look for our calibration files manually.

For the Bias images, fill the search parameters below with their associated
values and click on the ``Search`` button:

- Program ID: GN-CAL20170527-11
- Instrument: GMOS-N
- Binning: 1x1
- Raw / Reduced: Raw Only
- ROI: Full Frame

Once the page reloads, you should see a table with five files. The ``Type``
collumn will tell us that they are all BIAS. Go to the botton of the page and
click on the ``Download all 5 files totalling 0.06 Gb`` .

For the Flat images, fill the search form using the following parameters:

- Program ID: GN-CAL20170530-2
- Instrument: GMOS-N
- Binning: 1x1
- Raw / Reduced: Raw Only
- ROI: Full Frame

Now click on the little black arrow close to the ``Advanced Options`` and change
the ``QA State`` drop-down menu to **Pass** to ensure we have good quality data.

Press the ``Search`` button, the page will reload and show you six
files. The ``Type`` column says **OBJECT** but the ``Object`` columnn says
**Twilight**. This tells us that these are Twilight Flats. Go to the botton of
the page and click on the ``Download alll 6 files totalling 0.20 Gb`` .


Organize the data
=================

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