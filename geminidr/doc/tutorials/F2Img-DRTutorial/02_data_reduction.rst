
.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. _dataselect: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#dataselect

.. _descriptors: https://astrodata-user-manual.readthedocs.io/en/latest/appendices/appendix_descriptors.html

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

.. _setup_caldb:

Set up caldb_
-------------

DRAGONS comes with a local calibration manager and a local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive. This allows ``reduce`` to make requests for matching **processed**
calibrations when needed to reduce a dataset.

.. todo: calmanager
.. warning:: The Gemini Local Calibration Manager is not available yet in the
   Gemini Conda Channel for installation and you might not have it installed.
   On a terminal, type `caldb config`. If you get an error message, you can
   skip this section and you will still be able to bypass the Calibration
   Manager as we will show later here.


Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, create or edit the configuration file ``rsys.cfg`` as
follow:

.. code-block:: none

    [calibs]
    standalone = True
    database_dir = ${path_to_my_data}/f2img_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

.. note:: The tilde (``~``) in the path above refers to your home directory.
   Also, mind the dot in ``.geminidr``.

Then initialize the calibration database:

.. code-block:: bash

    caldb init

That's it! It is ready to use!

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file **only** from the database
(it will **not** remove the file on disk). For more the details, check the
`caldb documentation in the Recipe System - User's Manual <caldb>`_.


.. _check_files:

Check files
-----------

Now let us consider that we have put all the files in the same folder
called ``../playdata/`` and that we do not have any information anymore. From a
bash terminal and from within the Conda Virtual Environment where DRAGONS was
installed, we can call the command tool typewalk_:

.. code-block:: bash

   $ typewalk -d ../playdata/

   directory:  /path_to_my_files/f2img_tutorial/playdata
        S20131122S0439.fits ............... (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (F2) (GEMINI) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
        S20131127S0259.fits ............... (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (F2) (GEMINI) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
                     ...
        S20131127S0260.fits ............... (AT_ZENITH) (AZEL_TARGET) (CAL) (DARK) (F2) (GEMINI) (NON_SIDEREAL) (RAW) (SOUTH) (UNPREPARED)
        S20131121S0081.fits ............... (F2) (GEMINI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
   Done DataSpider.typewalk(..)

This command will open every FITS file within the folder passed after the ``-d``
flag (recursively) and will print an unsorted table with the file names and the
associated tags. For example, calibration files will always have the ``CAL``
tag. Flat images will always have the ``FLAT`` tag. Dark files will have the
``DARK`` tag. This means that we can start getting to know a bit more about our
data set just by looking the tags. The output above was trimmed for simplicity.


Create File lists
-----------------

This data set now contains science and calibration frames. It could have
different observed targets and different exposure times. The current data
reduction pipeline does not organize the data.

That means that we first need to identify these files and create lists that will
be used in the data-reduction process. For that, we will use the dataselect_
command line. Please, refer to the dataselect_ page for details regarding its
usage. Let us start with the DARK files:

.. code-block:: bash

   $ dataselect --tags DARK ../playdata/*.fits
   ../playdata/S20130930S0242.fits
   ../playdata/S20130930S0243.fits
                  ...
   ../playdata/S20140209S0544.fits
   ../playdata/S20140209S0545.fits

The ``--tags`` is a comma-separated argument that is used to select the files
that matches the tag(s) listed there.

Remember that our data set contains three sets of DARK files: one with 120 s
matching the science data, one with 20 s matching the flat data, and one
with 3 s to create BPMs. If you don't know what are the existing exposure times,
you can "pipe" the dataselect_ results and use the showd_ command line tool:

.. code-block:: bash

   $ dataselect --tags DARK ../playdata/*.fits | showd -d exposure_time
   -----------------------------------------------
   filename                          exposure_time
   -----------------------------------------------
   ../playdata/S20130930S0242.fits            20.0
   ../playdata/S20130930S0243.fits            20.0
                  ...
   ../playdata/S20131120S0115.fits           120.0
   ../playdata/S20131120S0116.fits           120.0
                  ...
   ../playdata/S20131127S0257.fits             3.0
   ../playdata/S20131127S0258.fits             3.0
                  ...
   ../playdata/S20140209S0544.fits            20.0
   ../playdata/S20140209S0545.fits            20.0

The ``|`` is what we call "pipe" and it is used to pass output from dataselect_
to showd_. The following line creates a list of DARK files that have exposure
time of 20 seconds:

.. code-block:: bash

   $ dataselect --tags DARK --expr "exposure_time==20" ../playdata/*.fits > darks_020s.list

``--expr`` is used to filter the files based on their descriptors_. Here we are
selecting files with exposure time of 20 seconds. You can repeat the same
command for the other existing exposure times (3 s and 120 s).

.. code-block:: bash

   $ dataselect --tags DARK --expr "exposure_time==3" ../playdata/*.fits > darks_003s.list

   $ dataselect --tags DARK --expr "exposure_time==120" ../playdata/*.fits > darks_120s.list

Now let us create the list containing the flat files:

.. code-block:: bash

    $ dataselect --tags FLAT ../playdata/*.fits > flats.list

We know that our dataset has only one filter (Y-band). If our dataset contains
data with more filters, we can use the ``--expr`` to select the appropriate
filter:

For that, we start creating the lists containing the
corresponding files for each filter:

.. code-block:: bash

    $ dataselect --tags FLAT --expr "filter_name=='Y'" ../playdata/*.fits > flats_Y.list

.. note::

    Remember that the FLAT images for Y, J and H must be taken with the
    instrument lamps on and off. This difference will be used during the
    creation of a master flat for each of these filters. For the Ks filter, only
    lamp off images are used.


Finally, we want to create a list with science targets. We are looking for files
whose are not calibration nor a bad-pixel-mask. To exclude them from our
selection we can use the ``--xtags``, e.g., ``--xtags CAL,BPM``.

.. code-block:: bash

    $ dataselect --xtags CAL,BPM ../playdata/*.fits > sci_images.list

Remember that you can use the ``--expr`` option to select targets with different
names or exposure times, depending on their descriptors_.


.. _process_dark_files:

Process DARK files
------------------

We start the data reduction by creating a Master Dark file for each exposure
time. We already created a list for each of them in the previous section and
now we can simply use the reduce_ command line to process them. Here is how
you can reduce the 20 s dark data and stack them into a Master Dark:

.. code-block:: bash

    $ reduce @darks_020s.list

Note the ``@`` character before the name of the file that contains the list of
DARKS. This syntax was inherited from IRAF and also works with most of DRAGONS
command line tools. More details can be found in the
`DRAGONS - Recipe System User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/howto.html#the-file-facility>`_.

Repeat the same commands for each exposure time and you will have a set of
MASTER Darks:

.. code-block:: bash

   $ reduce @darks_120s.list

   $ reduce @darks_003s.list

The Master DARK files will be saved in the same folder where reduce_ was
called and inside the ``./calibration/processed_dark`` folder. The former is
used to save cashed calibration files. If you have
`your local database configured <caldb>`_, you can add the Master DARK files to
it. This can be done using the following command:

.. code-block:: bash

    $ caldb add ./calibration/processed_dark/S20130930S0242_dark.fits

Remember that the filename can change. caldb_ only accepts **one file per
command**. If you want to add more files, you can repeat the command for each of
them.

.. code-block:: bash

   $ caldb add ./calibration/processed_dark/S20130930S0242_dark.fits

   $ caldb add ./calibration/processed_dark/S20131127S0257_dark.fits

Now reduce_ will be able to find these files if needed while processing other
data. If you have problems `setting up your local database <caldb>`_, you will
still be able to pass the files to reduce_ manually. This will be shown ahead
in this tutorial.

.. note::

    The DARK subtraction can be skipped sometimes. The two major situation that
    this can happen is when you have much more dithering frames on sky and when
    you have the same number of flats with LAMPON and LAMPOFF.


Create Bad-Pixel-Mask
---------------------

The Bad Pixel Mask (BPM) can be built using a set of flat images with the
lamps on and off and a set of short exposure dark files. Here, our shortest dark
files have 3 second exposure time. Again, we use the reduce_ command to
produce the BPMs.

It is important to note that **the recipe system only opens the first AD object
in the input file list**. So you need to send it a list of flats and darks, but
the _first_ file must be a flat. If the first file is a dark, then no, it won't
match that recipe.

Since Flamingos-2 filters are in the collimated space, the filter choice should
not interfere in the results.

.. code-block:: bash

    $ reduce @flats_Y.list @darks_003s.list -r makeProcessedBPM

Note that instead of creating a new list for the BP masks, we simply used a
flat list followed by the dark list. This ensures that the first file read by
reduce_ is a flat file. Also note the ``-r`` tells reduce_ to use a different
recipe instead of the default. The output image will be saved in the current
working directory and will have a ``_bpm`` suffix.


Process Flat-Field images
-------------------------

Master Flats can also be created using the reduce_ command line with the
default recipe.

.. code-block:: bash

    $ reduce @flats_Y.list -p addDQ:user_bpm="S20131129S0320_bpm.fits"

.. todo @bquint Review BPM injection
.. todo:: @bquint The command line above should pass the BPM to the ``p.addDQ``
   but it seems it is not. I am receiving ``WARNING - No static BPMs defined``
   messages while reducing the data. I checked with and without this option and
   I get the same message but the two masks are not the same.

Here, the ``-p`` argument tells reduce_ to modify the ``user_bpm`` in the
``addDQ`` primitive.

.. todo: calmanager

The following command can be used to bypass the calibration manager for the
case you don't have it or simply want to override the input calibrations:

.. code-block:: bash

   $ reduce @flats_Y.list -p addDQ:user_bpm=S20131129S0320_bpm.fits --user_cal processed_dark:S20130930S0242_dark.fits


Then, if you have your `local database configured <caldb>`_, we add the master
flat file to the database so reduce_ can find and use it when reducing the
science files.

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

Now that we have the Master Dark and Master Flat images, we can tell reduce_
to process our data. reduce_ will look at the remote or at the local database
for calibration files. Make sure that you have
`configured your database <caldb>`_ before running it. If you do not have a
local database, you still can pass the calibration files to reduce. This will
be shown later. For now, let us see the simplest case of reduce_:

.. code-block:: bash

    $ reduce @sci_images.list


This command will subtract the master dark and apply flat correction. Then it
will look for sky frames. If it does not find, it will use the science frames
and try to calculate sky frames using the dithered data. These sky frames will
be subtracted from the associated science data. Finally, the sky-subtracted
files will be stacked together in a single file.

.. todo: calmanager

The following command can be used to bypass the calibration manager for the
case you don't have it or simply want to override the input calibrations:

.. code-block::

   $ reduce @sci_images.list --user_cal processed_dark:S20131120S0116_dark.fits processed_flat:S20131129S0320_flat.fits

.. warning::

    The science exposures in all bands suffer from vignetting of the field in
    the NW quadrant (upper left in the image above). This may have been caused
    by the PWFS2 guide probe, which was used because of a hardware problem with
    the OIWFS (see the `F2 instrument status note <https://www.gemini.edu/sciops/instruments/flamingos2/status-and-availability>`_
    for 2013 Sep. 5). Therefore the photometry of this portion of the image will
    be seriously compromised.

The final product file will have a ``_stack.fits`` sufix and it is shown below:

.. the figure below can be created using the script inside the ``savefig``
   folder.

.. figure:: _static/S20131121S0075_stack.fits.png
   :align: center

.. todo @bquint Is this true?
.. todo:: @bquint Is this true?

If you passed the BPM when reducing the flats, it should be propagated to the
science data. If, for whatever reason, you did not pass the BPM before, you can
still do it now by using the ``-p`` as explained before:

.. code-block:: bash

   $ reduce @sci_images.list -p addDQ:user_bpm="S20131129S0320_bpm.fits"

Finally, you can pass the calibration files to reduce_ in the command line. This
is particularly useful if you have problems setting up your local database. This
can be done via ``--user_cal`` option:

.. code-block:: bash

   $ reduce @sci_images.list -p addDQ:user_bpm="S20131129S0320_bpm.fits"

.. todo @bquint How can I know that my calibration file is actually being used?
.. todo:: @bquint How can I know that my calibration file is actually being
   used?


