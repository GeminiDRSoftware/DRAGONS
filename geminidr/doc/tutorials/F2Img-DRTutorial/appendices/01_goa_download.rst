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

This tutorial uses observations from program GS-2013B-Q-15 (PI: Leggett),
NIR photometry of the faint T-dwarf star WISE J041358.14-475039.3, obtained on
2013-Nov-21. Images of this sparse field were obtained in the Y, J, H, Ks bands
using a dither sequence; daytime calibrations, darks and GCAL flats, were obtained as well.
`Leggett, et al. (2015) <https://ui.adsabs.harvard.edu/#abs/2015ApJ...799...37L/abstract>`_
briefly describes the data reduction procedures they followed, which are
similar to those described in this tutorial.

The first step of any reduction is to retrieve the data from the
`Gemini Observatory Archive (GOA) <https://archive.gemini.edu/>`_. For more
details on using the Archive, check its
`Help Page <https://archive.gemini.edu/help/index.html>`_.


Science data
------------

Navigate to the `GOA webpage <https://archive.gemini.edu/>`_ search form.  Put the data label
**GS-2013B-Q-15-39** in the ``PROGRAM ID`` text field, and press the ``Search``
button in the middle of the page. The page will refresh and display a table with
all the data for this dataset. Since the amount of data is unnecessarily large
for a tutorial (162 files, 0.95 GB), we have narrowed our search to only the
Y-band data by setting the ``Instrument`` drop-down menu to **F2** and the
``Filter`` drop-down menu to **Y**. Now we have only 9 files, 0.05 GB.

You can also copy the URL below and paste it in a browser to see the search
results:

::

  https://archive.gemini.edu/searchform/GS-2013B-Q-15-39/RAW/cols=CTOWEQ/filter=Y/notengineering/F2/NotFail

At the bottom of the page, you will find a button saying *Download all 9 files
totalling 0.05 GB*. Click on it to download a `.tar` file with all the data.

Calibrations
------------
Matching calibration files can be obtained by clicking on the *Load Associated
Calibrations* tab. For this data, we need the 120-second darks (for 120-second
science data). We also need the Y-band flats; the series there is a collection
of lamp-on and lamp-off flats.

Select the darks and the Y-band flats at the top of the returned list by
checking the little boxes on the left. Scroll down and click "Download
Marked Files"

Finally, you will need a set of short dark frames in order to create the Bad
Pixel Masks (BPM). For that, we will have to perform a search ourselves in the
archive.

First remove the Program ID. The science data was obtained on November 21,
2013. So, we set the "UTC Date" to a range of a few days around the
observations date. This and other settings are:

- Program ID: <empty>
- UTC Date: 20131120-20131122
- Instrument: F2
- Obs. Type: Dark
- Filter: Any

Hit the "Search" button. You can sort the list by exposure time by clicking
on the header of the "ExpT" column. Several 2-second darks show up. Some were
even taken on the same date as the science data (20131121). Select those,
and download them as we did before for the other calibrations.


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
     and create a parallel directory of running the tutorial called
     ``playground``. The tutorial makes assumption as to where everything
     is located.
