
.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. _dataselect: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#dataselect

.. _reduce: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#typewalk

.. _showd: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showd

.. _show_primitives: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#show-primitives

.. _show_recipes: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#show-recipes

.. _showpars: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showpars

.. _typewalk: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#typewalk


.. _command_line_data_reduction:

Data Reduction
**************

This chapter will guide you on reducing **Flamingos-2 Images** using command
line tools.

DRAGONS installation comes with a set of handful scripts that are used to
reduce astronomical data. One of the most important scripts is called
reduce_, which is extensively explained in the `Recipe System Users Manual
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/index.html>`_.
For this tutorial, we will be also using other `Supplemental tools
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html>`_,
like dataselect_, showd_, typewalk_, and caldb_.

.. warning:: Some primitives use a lot of RAM memory and they can make `reduce`
    crash. Our team is aware of this problem and we are working on that. For
    now, if that happens to you, you might need to run the pipeline on a
    smaller data set.

Process DARK files
------------------

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
`reduce`_ command line to reduce and stack them into a single Master DARK file:

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

And then pass this list to the `reduce`_ command.

.. code-block:: bash

    $ reduce @darks_120s.list

The Master DARK files will be saved in the same folder where `reduce`_ was
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


Now `reduce`_ will be able to find these files if needed while processing other
data.

.. note::

    The DARK subtraction can be skipped sometimes. The two major situation that
    this can happen is when you have much more dithering frames on sky and when
    you have the same number of flats with LAMPON and LAMPOFF.


Create Bad-Pixel-Mask
---------------------

The Bad Pixel Mask (BPM) can be built using a set of flat images with the
lamps on and off and a set of short exposure dark files. Here, our shortest dark
files have 3 second exposure time. Again, we use the `reduce`_ command to
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
flat list followed by the dark list. Note also the ``-r`` tells `reduce`_ to
use a different recipe instead of the default.


Process Flat-Field images
-------------------------

Master Flats can also be created using the `reduce`_ command line with the
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


Here, the ``-p`` argument tells `reduce`_ to modify the ``user_bpm`` in the ``addDQ``
primitive. Then, we add the master flat file to the database so `reduce`_ can
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
---------------------

Now that we have the Master Dark and Master Flat images, we can tell `reduce`_
to process our data. `reduce`_ will look at the remote or at the local database
for calibration files. Make sure that you have `configured your database <caldb>`_
before running it. We want to run `reduce`_ on any file that is not calibration
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
