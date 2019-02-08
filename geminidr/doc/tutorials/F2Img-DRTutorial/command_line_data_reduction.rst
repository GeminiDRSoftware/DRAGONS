
.. _command_line_data_reduction:

Reducing F2 Images via Command Line
===================================

Before start, make sure you have:

    - Anaconda is properly installed;
    - A Virtual Environment was properly created and is active;
    - AstroConda (STScI) is properly installed within the Virtual Environment;
    - DRAGONS was successfully installed within the Virtual Environment;

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

This tutorial is written based on the `Flamingos 2 Cookbook <http://rashaw-science.org/F2_drc/>`_
and will use the same data sets for Imaging and Long-Slit Spectroscopy.

It reduces images from the program GS-2013B-Q-15 (PI: Leggett), A Study of the
450K Transition from T to Y Dwarf, and of the 350K Y Dwarfs. This team obtained
images with Y, J, H, Ks filters of the T9-dwarf star WISE J041358.14-475039.3.
See
`Leggett et al. (2015) <https://ui.adsabs.harvard.edu/#abs/2015ApJ...799...37L/abstract>`_
for details of the science objectives and their data reduction procedure.


Retrieve and Organize Data
--------------------------
This tutorial will use observations from program GS-2013B-Q-15 (PI:Leggett),
NIR photometry of the faint T-dwarf star WISE J041358.14-475039.3, obtained on
2013-Nov-21. Images of this sparse field were obtained in the Y, J, H, Ks bands
using a dither sequence; dayCal darks and GCAL flats were obtained as well.
Leggett, et al. (2015; [L15]) briefly describes the data reduction procedures
they followed, which are similar to those described below.

The first step is to retrieve the data from the Gemini Observatory Archive (see
`Archive Searches <http://rashaw-science.org/F2_drc/GettingStarted.html#archive-search>`_).

The full data set contains 330 images and uses a total of 5.3 Gb of disk space.
Each raw and unprocessed image has 17 Mb but some processed images have more.
The data reduction steps are almost the same for all filters so, for this tutorial,
we will use a sub data set that contains only the images obtained with the Y
filter and its calibrations. This sub-set has only 49 files and 786 Mb of disk
space.

We now need retrieve exposures within about a month of the target exposures in
2013 Nov 21. You may search the `GOA <https://archive.gemini.edu/searchform>`_
yourself, or instead just cut-and-paste the following direct URL in your
browser.

::

   # images of the WISE 0413-4750 target field:
   https://archive.gemini.edu/searchform/GS-2013B-Q-15-39/RAW/cols=CTOWEQ/NOTAO/filter=Y/notengineering/F2/imaging/20130101-20150701/AnyQA#


After retrieving the science data, click the Load Associated Calibrations tab on
the search results page and download the associated dark and flat-field
exposures.

Unpack all of them in a subdirectory of your working directory (assumed to be
named /raw in this tutorial). Be sure to uncompress the files. See
`Retrieving Data <http://rashaw-science.org/F2_drc/GettingStarted.html#retrieve-data>`_
for details.

.. Exposure Summary
   ----------------
   The data contain exposures of a specific science target and
   `dayCal <http://rashaw-science.org/F2_drc/Glossary.html#term-daycal>`_
   calibrations; see the table below for a summary. All exposures were obtained
   with ``ReadMode = Bright``. The science exposures were obtained in a
   :math:`3 \times 3` spatial dither pattern, with a spacing of about 15 arcsec in
   each direction from the initial alignment (see
   `IR Background Removal <http://rashaw-science.org/F2_drc/Supplement.html#ir-background>`_).

   ================ ======== =============== =====================
    Target           Filter   Exposure Time   Number of Exposures
   ================ ======== =============== =====================
    WISE 0413-4750   Y        120 s           9
    ...              J        60 s            9
    ...              H        15 s            72
    ...              Ks       15 s            72
    Dark             ...      120 s           10
    ...              ...      60 s            21
    ...              ...      20 s            20
    ...              ...      15 s            10
    ...              ...      8 s             25
    ...              ...      3 s             13
    GCAL Flat	      Y        20 s            4 (on) / 6 (off)
    ...              J        60 s            4 (on) / 6 (off)
    ...              H        3 s             4 (on) / 6 (off)
    ...              Ks       8 s             12 (off)
   ================ ======== =============== =====================

   The GCAL exposures list those for
   `Lamps-On <http://rashaw-science.org/F2_drc/Glossary.html#term-lamps-on>`_ and
   `Lamps-Off <http://rashaw-science.org/F2_drc/Glossary.html#term-lamps-off>`_
   separately. The exposure duratlsions above are noted in the ``obsConfig.yml`` file.
   We will use calibration exposures obtained within a few days of the observations.

DARK MasterCal
--------------

We usually start our data reduction by stacking the DARKS files that contains
the same exposure time into a Master DARK file. Before we do that, let us create
file lists that will be used later. These lists can be easily produced using the
|dataselect| command line, which is available after installing DRAGONS in your
virtual environment.

.. todo::

   add `dataselect` documentation somewhere.

We start with the creating of lists that contains DARK images for every exposure
time available. If you don't know what are the existing exposure times, you can
"pipe" the |dataselect| results and use the |showd| command line tool:

::

    (my_venv) $ dataselect --tags DARK --xtags PROCESSED raw/*.fits |
        showd -d exposure_time

The following line creates a list of DARK files that were not processed and that
have exposure time of 20 seconds:

::

   (my_venv) $ dataselect --tags DARK --xtags PROCESSED \
       --expr "exposure_time==20" raw/*.fits > darks_020s.list

The `--tags` is a comma-separated argument that is used to select the files
that matches the tag(s) listed there. The `--xtags` is used to exclude
the files which tags matches the one(s) listed here. The `--expr` is used
to filter the files based on their attributes. In this case, we are selecting
files with exposure time of 20 seconds. Use `dataselect --help` for more
information.

Once we have the list of DARK files for each exposure time, we can use the
`reduce` command line to reduce and stack them into a single Master DARK file:

::

    (my_venv) $ reduce @darks_020s.list

Note the `@` character before the name of the file that contains the list of
DARKS. This syntax was inherited from IRAF and also works with most of DRAGONS
command line tools. More details can be found in the
`DRAGONS - Recipe System User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/howto.html#the-file-facility>`_.

Repeat the same commands for each exposure time and you will have a set of
MASTER Darks. Again, we first create a list that contains the DARK files with
same exposure times:

::

    (my_venv) $ dataselect --tags DARK --xtags PROCESSED \
        --expr "exposure_time==120" raw/*.fits > darks_120s.list

And then pass this list to the `reduce` command.

::

    (my_venv) $ reduce @darks_120s.list



The Master DARK files will be saved in the same folder where `reduce` was called
and inside the `./calibration/processed_dark` folder. The former is used to save
cashed calibration files. If you have |your local database configured|, you
can add the Master DARK files to it. The following command is a shell trick
that will go over all the files that ends with `_dark.fits` and add them to
the database.

::

    (my_env) $ for f in `ls *_dark.fits`; do caldb add ${f}; done

Now `reduce` will be able to find these files if needed while processing other
data.

.. note::

    The DARK subtraction can be skipped sometimes. The two major situation that
    this can happen is when you have much more dithering frames on sky and when
    you have the same number of flats with LAMPON and LAMPOFF.


Bad-Pixel-Mask MasterCal
------------------------

The Bad Pixel Mask (BPM) can be built using a set of flat images with the
lamps on and off and a set of short exposure dark files. Here, our shortest dark
files have 20 second exposure time. Again, we use the `reduce` command to produce
the BPMs.

It is important to note that the recipe system only opens the first AD object in
the input file list. So you need to send it a list of flats and darks, but the
_first_ file must be a flat. If the first file is a dark, then no, it won't
match that recipe.

Since Flamingos-2 filters are in the collimates space, the filter choice should
not interfere in the results.

::

    (my_env) $ dataselect --tags FLAT --xtags PREPARED \
        --expr "filter_name=='Y'" *.fits > flats_Y.list
    (my_env) $ reduce @flats_Y.list @darks_020s.list -r makeProcessedBPM

Note that instead of creating a new list for the BP masks, we simply used a
flat list followed by the dark list. Note also the `-r` tells `reduce` to use a
different recipe instead of the default.


Flat-Field MasterCal
--------------------

Master Flats can also be created using the `reduce` command line with the default
recipe. For that, we start creating the lists containing the corresponding files
for each filter:

::

    (my_env) $ dataselect --tags FLAT --xtags PREPARED \
        --expr "filter_name=='Y'" *.fits > flats_Y.list

.. note::

    Remember that the FLAT images for Y, J and H must be taken with the
    instrument lamps on and off. This difference will be used during the
    creation of a master flat for each of these filters. For the Ks filter, only
    lamp off images are used.

::

    (my_env) $ reduce @flats_Y.list -p addDQ:user_bpm="S20131129S0320_bpm.fits"

Here, the `-p` argument tells `reduce` to modify the `user_bpm` in the `addDQ`
primitive. Then, we add the master flat file to the database so `reduce` can
find and use it when reducing the science files.

::

    (my_env) $ caldb add S20131129S0320_flat.fits

.. note::

    The Ks-band thermal emission from the GCAL shutter depends upon the
    temperature at the time of the exposure, and includes some spatial
    structure. Therefore the distribution of emission is not necessarily
    consistent, except for sequential exposures. So it is best to combine
    lamps-off exposures from a single day.

Reducing Science Images
-----------------------

Now that we have the Master Dark and Master Flat images, we can tell `reduce`
to process our data. `reduce` will look at the remote or at the local database
for calibration files. Make sure that you have |configured your database|
before running it. We want to run `reduce` on any file that is not calibration
nor a bad-pixel-mask (`--xtags CAL,BPM`). We also want to run this pipeline
only on Y band images (`--expr 'filter_name=="Y"'`)

::

    (my_env) $ dataselect --xtags CAL,BPM --expr 'filter_name=="Y"' \
        raw/*.fits > sci_images_Y.list
    (my_env) $ reduce @sci_images_Y.list

This command will subtract the master dark and apply flat correction. Then it
will look for sky frames. If it does not find, it will use the science frames
and try to calculate sky frames using the dithered data. These sky frames will
be subtracted from the associated science data. Finally, the sky-subtracted
files will be stacked together in a single file. The final result is shown
below:

.. figure:: _static/S20131121S0075_stack.fits.png
   :align: center

   S20131121S0075_stack.fits.png

.. warning::

    The science exposures in all bands suffer from vignetting of the field in
    the NW quadrant (upper left in the image above). This may have been caused
    by the PWFS2 guide probe, which was used because of a hardware problem with
    the OIWFS (see the F2 instrument status note for 2013 Sep. 5). Therefore the
    photometry of this portion of the image will be seriously compromised.