
.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. _dataselect: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#dataselect

.. _showd: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showd


.. _data_reduction:

**************
Data Reduction
**************

This chapter will guide you on reducing **Flamingos-2 Images**. Before we start,
make sure you have:

    - Anaconda is properly installed;
    - A Conda Environment was properly created and is active;
    - AstroConda (STScI) is properly installed within the Conda Environment;
    - DRAGONS was successfully installed within the Conda Environment;

DRAGONS installation comes with a set of handful scripts that are used to
reduce astronomical data. One of the most important scripts is called
``reduce``, which is extensively explained in the
`Recipe System Users Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/index.html>`_.
For this tutorial, we will be also using other
`Supplemental tools <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html>`_.

.. todo::
    Add `caldb` to the supplemental tools

.. todo::
    Add `dataselect` to the supplemental tools

.. todo::
    Add `typewalk` to the supplemental tools

.. todo::
    Add `showd` to the supplemental tools


Retrieve and Organize Data
==========================

This tutorial will use observations from program GS-2013B-Q-15 (PI:Leggett),
NIR photometry of the faint T-dwarf star WISE J041358.14-475039.3, obtained on
2013-Nov-21. Images of this sparse field were obtained in the Y, J, H, Ks bands
using a dither sequence; dayCal DARKS and GCAL flats were obtained as well.
Leggett, et al. (2015; `[L15]
<https://ui.adsabs.harvard.edu/#abs/2015ApJ...799...37L/abstract>`_)
briefly describes the data reduction procedures they followed, which are
similar to those described below.

The first step is to retrieve the data from the Gemini Observatory Archive
(GOA). For more details on using the Archive, check its
`Help Page <https://archive.gemini.edu/help/index.html>`_. The link below takes
you to the result obtained when searching for data that corresponds to the
chosen program.

::

   https://archive.gemini.edu/searchform/GS-2013B-Q-15-39

The bottom of the page contains a button to download the data. You can click on
that, or you can download the images by `clicking directly
here <https://archive.gemini.edu/download/GS-2013B-Q-15-39/present/canonical>`_.
Alternatively, you can download the data by copy-and-pasting the address below
in your browser:

::

   https://archive.gemini.edu/download/GS-2013B-Q-15-39/present/canonical

After retrieving the science data, click the Load Associated Calibrations tab on
the search results page and download the associated dark and flat-field
exposures. Again, the calibration files can be downloaded by `clicking here
<https://archive.gemini.edu/download/associated_calibrations/GS-2013B-Q-15-39/canonical>`_
or by copying the following URL to your browser:

::

    https://archive.gemini.edu/download/associated_calibrations/GS-2013B-Q-15-39/canonical

Unpack all of them in a subdirectory of your working directory (assumed to be
named /raw in this tutorial). Be sure to uncompress the files.

.. code-block:: bash

   $ cd <my_main_working_directory>

   $ tar -xvf *calib*.tar # extract calibration files from .TAR file

   $ tar -xvf *data*.tar # extract science files from .TAR file

   $ bunzip2 *.fits.bz2 # command that will decompress FITS files

   $ mkdir raw/ # create directory named "raw" (optional)

   $ mv *.fits raw/ # move all the raw FITS files to raw (optional)

The full de-compressed data set will have 310 files and use 4.9 Gb of disk
space.


Process DARK files
==================

We usually start our data reduction by stacking the DARKS files that contains
the same exposure time into a Master DARK file. Before we do that, let us create
file lists that will be used later. These lists can be easily produced using the
dataselect_ command line, which is available after installing DRAGONS in your
conda environment.

We start with the creating of lists that contains DARK images for every exposure
time available. If you don't know what are the existing exposure times, you can
"pipe" the dataselect_ results and use the showd_ command line tool:

.. code-block:: bash

    $ dataselect --tags DARK --xtags PROCESSED raw/*.fits | showd -d exposure_time

The ``|`` is what we call "pipe" and it is used to pass output from dataselect_
to showd_. The following line creates a list of DARK files that were not
processed and that have exposure time of 20 seconds:

.. code-block:: bash

   $ dataselect --tags DARK --xtags PROCESSED \
       --expr "exposure_time==20" raw/*.fits > darks_020s.list

The ``\`` is simply a special character to break the line. The ``--tags`` is a
comma-separated argument that is used to select the files that matches the
tag(s) listed there. ``--xtags`` is used to exclude the files which tags
matches the one(s) listed here. ``--expr`` is used to filter the files based
on their attributes. Here we are selecting files with exposure time of
20 seconds. You can repeat the same command for the other existing exposure
times (3 s, 8 s, 15 s, 60 s, 120 s). Use ``dataselect --help`` for more
information.

Once we have the list of DARK files for each exposure time, we can use the
``reduce`` command line to reduce and stack them into a single Master DARK file:

.. code-block:: bash

    $ reduce @darks_020s.list

Note the ``@`` character before the name of the file that contains the list of
DARKS. This syntax was inherited from IRAF and also works with most of DRAGONS
command line tools. More details can be found in the
`DRAGONS - Recipe System User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/howto.html#the-file-facility>`_.

Repeat the same commands for each exposure time and you will have a set of
MASTER Darks. Again, we first create a list that contains the DARK files with
same exposure times:

.. code-block:: bash

    $ dataselect --tags DARK --xtags PROCESSED \
        --expr "exposure_time==120" raw/*.fits > darks_120s.list

And then pass this list to the ``reduce`` command.

.. code-block:: bash

    $ reduce @darks_120s.list

The Master DARK files will be saved in the same folder where ``reduce`` was
called and inside the ``./calibration/processed_dark`` folder. The former is
used to save cashed calibration files. If you have
`your local database configured <caldb>`_, you can add the Master DARK files to
it. This can be done using the following command:

.. code-block:: bash

    $ caldb add ./calibration/processed_dark/S20130930S0242_dark.fits

`caldb`_ only accepts **one file per command**. If you want to add more files,
you can repeat the command for each of them.

.. tip::

    For those that do not want to repeat the same command again and again,
    use
    ``$ for f in `ls calibrations/processed_dark/*_dark.fits`; do caldb add ${f}; done``
    bash command. It will list the files that end with ``*_dark.fits`` and
    add them to the ``caldb`` one by one.


Now ``reduce`` will be able to find these files if needed while processing other
data.

.. note::

    The DARK subtraction can be skipped sometimes. The two major situation that
    this can happen is when you have much more dithering frames on sky and when
    you have the same number of flats with LAMPON and LAMPOFF.


Create Bad-Pixel-Mask
=====================

The Bad Pixel Mask (BPM) can be built using a set of flat images with the
lamps on and off and a set of short exposure dark files. Here, our shortest dark
files have 3 second exposure time. Again, we use the ``reduce`` command to
produce the BPMs.

It is important to note that **the recipe system only opens the first AD object
in the input file list**. So you need to send it a list of flats and darks, but
the _first_ file must be a flat. If the first file is a dark, then no, it won't
match that recipe.

Since Flamingos-2 filters are in the collimated space, the filter choice should
not interfere in the results.

.. code-block:: bash

    $ dataselect --tags FLAT --xtags PREPARED \
        --expr "filter_name=='Y'" raw/*.fits > flats_Y.list

    $ reduce @flats_Y.list @darks_003s.list -r makeProcessedBPM

Note that instead of creating a new list for the BP masks, we simply used a
flat list followed by the dark list. Note also the ``-r`` tells ``reduce`` to
use a different recipe instead of the default.


Process Flat-Field images
=========================

Master Flats can also be created using the ``reduce`` command line with the
default recipe. For that, we start creating the lists containing the
corresponding files for each filter:

.. code-block:: bash

    $ dataselect --tags FLAT --xtags PREPARED \
        --expr "filter_name=='Y'" raw/*.fits > flats_Y.list


.. note::

    Remember that the FLAT images for Y, J and H must be taken with the
    instrument lamps on and off. This difference will be used during the
    creation of a master flat for each of these filters. For the Ks filter, only
    lamp off images are used.


.. code-block:: bash

    $ reduce @flats_Y.list -p addDQ:user_bpm="S20131129S0320_bpm.fits"


Here, the ``-p`` argument tells ``reduce`` to modify the ``user_bpm`` in the ``addDQ``
primitive. Then, we add the master flat file to the database so ``reduce`` can
find and use it when reducing the science files.

.. code-block:: bash

    $ caldb add ./calibrations/processed_flat/S20131129S0320_flat.fits


.. warning::

    The Ks-band thermal emission from the GCAL shutter depends upon the
    temperature at the time of the exposure, and includes some spatial
    structure. Therefore the distribution of emission is not necessarily
    consistent, except for sequential exposures. So it is best to combine
    lamps-off exposures from a single day.


Reduce Science Images
=====================

Now that we have the Master Dark and Master Flat images, we can tell ``reduce``
to process our data. ``reduce`` will look at the remote or at the local database
for calibration files. Make sure that you have `configured your database <caldb>`_
before running it. We want to run ``reduce`` on any file that is not calibration
nor a bad-pixel-mask (``--xtags CAL,BPM``). We also want to run this pipeline
only on Y band images (``--expr 'filter_name=="Y"'``)

.. code-block:: bash

    $ dataselect --xtags CAL,BPM --expr 'filter_name=="Y"' \
        raw/*.fits > sci_images_Y.list

    $ reduce @sci_images_Y.list


This command will subtract the master dark and apply flat correction. Then it
will look for sky frames. If it does not find, it will use the science frames
and try to calculate sky frames using the dithered data. These sky frames will
be subtracted from the associated science data. Finally, the sky-subtracted
files will be stacked together in a single file.

.. warning::

    The science exposures in all bands suffer from vignetting of the field in
    the NW quadrant (upper left in the image above). This may have been caused
    by the PWFS2 guide probe, which was used because of a hardware problem with
    the OIWFS (see the F2 instrument status note for 2013 Sep. 5). Therefore the
    photometry of this portion of the image will be seriously compromised.

The final product file will have a ``_stack.fits`` sufix and it is shown below:

.. figure:: _static/S20131121S0075_stack.fits.png
   :align: center

   S20131121S0075_stack.fits.png
