.. 02_data_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/caldb.html

.. _data quality plane: https://astrodata-user-manual.readthedocs.io/en/latest/data.html#data-quality-plane

.. _dataselect: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#dataselect

.. _descriptors: https://astrodata-user-manual.readthedocs.io/en/latest/appendices/appendix_descriptors.html

.. _reduce: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/reduce.html

.. _showd: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#showd

.. _show_primitives: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#show-primitives

.. _show_recipes: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#show-recipes

.. _typewalk: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#typewalk


.. _command_line_data_reduction:

**************
Data Reduction
**************

This chapter will guide you on reducing **GMOS imaging data** using
command line tools. In this example we reduce a GMOS observation of
a star and distant galaxy field. The observation is a simple dither-on-target
sequence. Just open a terminal to get started.

While the example cannot possibly cover all situations, it will help you get
acquainted with the reduction of GMOS data with DRAGONS. We encourage you to
look at the :ref:`tips_and_tricks` and :ref:`issues_and_limitations` chapters to
learn more about GMOS data reduction.

DRAGONS installation comes with a set of handful scripts that are used to
reduce astronomical data. The most important script is called
reduce_, which is extensively explained in the `Recipe System Users Manual
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/index.html>`_.
It is through that command that a DRAGONS reduction is launched.

For this tutorial, we will be also using other `Supplemental tools
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html>`_,
like:

* dataselect_
* showd_
* typewalk_
* caldb_

The dataset
===========

If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`about_data_set`.

Here is a copy of the table for quick reference.

+---------------+---------------------+--------------------------------+
| Science       || N20170525S0116-120 | 300 s, g-band                  |
+---------------+---------------------+--------------------------------+
| Bias          || N20170527S0528-532 |                                |
+---------------+---------------------+--------------------------------+
| Twilight Flats|| N20170530S0360     | 256 s, g-band                  |
|               || N20170530S0363     | 64 s, g-band                   |
|               || N20170530S0364     | 32 s, g-band                   |
|               || N20170530S0365     | 16 s, g-band                   |
|               || N20170530S0371-372 | 1 s, g-band                    |
+---------------+---------------------+--------------------------------+


.. _setup_caldb:

Set up caldb_
=============

DRAGONS comes with a local calibration manager and a local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive. This allows ``reduce`` to make requests for matching **processed**
calibrations when needed to reduce a dataset. If you have problems setting up
caldb_ or want to bypass it for another reason, you can check the
`Bypassing automatic calibration association <bypassing_caldb>`_ section.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, create or edit the configuration file ``rsys.cfg`` as
follow:

.. code-block:: none

    [calibs]
    standalone = True
    database_dir = /path_to_my_data/gsaoiimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

..  note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.geminidr``.

..  todo:: @dragons_team The command below should give a feedback to the user.
    How do I know that the command actually worked? How do I know that I
    am using the correct database?

Then initialize the calibration database:

.. code-block:: bash

    caldb init

That's it! It is ready to use!

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file **only** from the database
(it will **not** remove the file on disk). For more the details, check the
caldb_ documentation in the
`Recipe System: User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/>`_.


.. _check_files:

Check files
===========

For this example, all the raw files we need are in the same directory called
``../playdata/``. Let us learn a bit about the data we have.

Ensure that you are in the ``playground`` directory and that the ``conda``
environment that includes DRAGONS has been activated.

Let us call the command tool typewalk_:

..  code-block:: bash

    $ typewalk -d ../playdata/

    directory:  /data/bquint/tutorials/gmosimg_tutorial/playdata
         N20170525S0116.fits ............... (GEMINI) (GMOS) (IMAGE) (NORTH) (RAW) (SIDEREAL) (UNPREPARED)
         ...
         N20170527S0528.fits ............... (AT_ZENITH) (AZEL_TARGET) (BIAS) (CAL) (GEMINI) (GMOS) (LS) (NON_SIDEREAL) (NORTH) (RAW) (UNPREPARED)
         ...
         N20170530S0360.fits ............... (CAL) (FLAT) (GEMINI) (GMOS) (IMAGE) (NORTH) (RAW) (SIDEREAL) (TWILIGHT) (UNPREPARED)
         ...
    Done DataSpider.typewalk(..)


This command will open every FITS file within the folder passed after the ``-d``
flag (recursively) and will print an unsorted table with the file names and the
associated tags. For example, calibration files will always have the ``CAL``
tag. Flat images will always have the ``FLAT`` tag. This means that we can start
getting to know a bit more about our data set just by looking the tags. The
output above was trimmed for simplicity.


.. _create_file_lists:

Create File lists
=================

This data set contains science and calibration frames. For some programs, it
could have different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you. You
have to do it. DRAGONS provides tools to help you with that.

The first step is to create lists that will be used in the data reduction
process. For that, we use dataselect_. Please, refer to the dataselect_
documentation for details regarding its usage.

List of Bias
------------

Our data set contains a set of BIAS files. You can select the BIAS filas using
dataselect_ and pass it to a file using the ``>`` symbol:

..  code-block:: bash

    $ dataselect --tags BIAS ../playdata/*.fits > list_of_bias.txt

If you want to see the output, simply omit ``>`` and the filename.


List of Flats
-------------

Now we can do the same with the FLAT files:

..  code-block:: bash

    $ dataselect --tags FLAT ../playdata/*.fits > list_of_flats.txt


If your dataset has FLATs obtained with more than one filter, you can add the
``--expr 'filter_name=="g"'`` expression to get on the FLATs obtained with in
the g-band. For example:

.. code-block:: bash

    $ dataselect --tags FLAT --expr 'filter_name=="g"' ../playdata/*.fits > list_of_g-band_flats.txt


List for science data
---------------------

The rest is the data with your science target. The simplest way of creating a
list of science frames is excluding everything that is a calibration:

.. code-block:: bash

    $ dataselect --xtags CAL ../playdata/*.fits > list_of_science.txt


This will work for our dataset because we know that a single target was observed
with a single filter and with the same exposure time. But what if we don't knwo
that?

We can check it by passing the dataselect_ output to the showd_ command line
using a "pipe" (``|``):

..  code-block:: bash

    $ dataselect --expr 'observation_class=="science"' ../playdata/*.fits | showd -d object,exposure_time
    --------------------------------------------------------
    filename                          object   exposure_time
    --------------------------------------------------------
    ../playdata/N20170525S0116.fits    o3e43           300.0
    ../playdata/N20170525S0117.fits    o3e43           300.0
    ../playdata/N20170525S0118.fits    o3e43           300.0
    ../playdata/N20170525S0119.fits    o3e43           300.0
    ../playdata/N20170525S0120.fits    o3e43           300.0


The ``-d`` flag tells showd_ which descriptors_ will be printed for each input
file. As you can see, we have only observed target and only exposure time.

If you see more than one object, you can create a list for each standard star
using the ``object`` descriptor as an argument for dataselect_ (spaces are
allowed if you use double quotes):

.. code-block:: bash

    $ dataselect --expr 'object=="o3e43"' ../playdata/*.fits > list_of_sci_o3e43.txt


Now let us consider that we want to filter all the files whose ``object`` is
**o3e43** and that the ``exposure_time`` is **300 seconds**. We also want to
pass the output to a new list:

.. code-block:: bash

   $ dataselect --expr '(object=="o3e43" and exposure_time==300.)' ../playdata/*.fits > list_of_science_files.txt


We have our input lists and we have initialized the calibration database, we
are ready to reduce the data.

Please make sure that you are still in the ``playground`` directory.


.. _make_master_bias:

Make Master Bias
================

We start the data reduction by creating a master bias for the science data.
It can be created using the command below:

..  code-block:: bash

   $ reduce @list_of_bias.txt


Note the ``@`` character before the name of the input file. This is the
"at-file" syntax. More details can be found in the
`DRAGONS - Recipe System User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/howto.html#the-file-facility>`_.

Master Bias files can be added to the local database using the caldb_
command. Before you run it, make sure you have `configured and initialized your
caldb <setup_caldb>`_. Once you are set, add the Master Bias to the local
database using the following command:

..  code-block:: bash

    $ caldb add N20170527S0528_bias.fits

.. note::
    The master bias will be saved in the same folder where reduce_ was
    called *and* inside the ``./calibration/processed_bias`` folder. The latter
    location is to cache a copy of the file. This applies to all the processed
    calibration, eg. master flat.

    Some people might prefer adding the copy in the `calibration` directory
    as it is safe from a `rm *`, for example.

    .. code-block:: bash

        $ caldb add ./calibration/processed_dark/N20170527S0528_bias.fits

.. note::
    reduce_ uses the first filename in the input list as basename and adds
    ``_bias`` as a suffix to it. So if your first filename is, for example,
    ``N20001231S001.fits``, the output will be ``N20001231S001_bias.fits``. Because
    of that, the base name of the Master Bias file can be different for you.

Before carrying on, check that the Master Bias was added to the database
using the ``caldb list`` command:

.. code-block:: bash

    $ caldb list
    N20170527S0528_bias.fits       ${path_to_my_data}/playground/calibrations/processed_bias


.. _process_flat_files:

Make Master Flat
================

FLAT images can be easily reduced using the ``reduce`` command line:

..  code-block:: bash

   $ reduce @list_of_flats.txt


Note reduce_ will query the local calibration manager for the Master Bias frame
and use it in the data reduction.

Once finished you will have the Master Flat in the current work directory and
inside ``./calibrations/processed_flat``. It will have a ``_flat`` suffix.

Add the Master Flat to the local calibration database with the following
command:

..  code-block:: bash

    $ caldb add N20170530S0360_flat.fits

Again, check that the Master Flat was added to your local database:

.. code-block:: bash

  $ caldb list
  N20170527S0528_bias.fits       ${path_to_my_data}/playground/calibrations/processed_bias
  N20170530S0360_flat.fits       ${path_to_my_data}/playground/calibrations/processed_flat


.. _process_fringe_frame:

Make Master Fringe Frame
========================

.. note:: The dataset used in this tutorial does not require Fringe Correction
    so you can skip this section if you are following it. Find more information
    below.

The reduction of some datasets requires a Master Fringe Frame. The datasets
that need a Fringe Frame are shown in the appendix
`Fringe Correction Tables <fringe_correction_tables>`_.

If you find out that your dataset needs Fringe Correction, you can use the
command below to create the Master Fringe Frame:

.. code-block:: bash

    $ reduce @list_of_science.txt -r makeProcessedFringe

This command line will produce an image with the ``_fringe`` suffix in the
current working directory.

Once you have the, you still need to add it to the local calibration manager
database:

.. code-block:: bash

    $ caldb add N20170525S0116_fringe.fits

Again, note that this step is only needed for images obtained with some
detector and filter combinations. Make sure you checked the
`Fringe Correction Tables <fringe_correction_tables>`_.


.. _processing_science_files:

Reduce Science files
====================

Once we have our calibration files processed and added to the database, we can
run ``reduce`` on our science data:

.. code-block:: bash

   $ reduce @list_of_science.txt

This command will generate flat corrected and sky subtracted files and will
stack them. This stacked image will have the ``_stack`` suffix. You might see
some warning messages from AstroPy that are related to the header of the images.
It should be safe to ignore them for now.

Here is one of the raw images:

.. figure:: _static/img/N20170525S0116.png
   :align: center

   One of the multi-extensions files.

Once reduce_ runs, it adds a `data quality plane`_ with information about why
the data is being rejected. The figure below shows the reduced staked data and
the bad pixel mask (in light gray):

.. figure:: _static/img/N20170525S0116_stack.png
   :align: center

   Sky Subtracted and Stacked Final Image. The light-gray area represents the
   masked pixels.

The DQ plane is updated on every data reduction step and most of the
calculations are done on the good data. Because of this, you might expect to see
some leftover features if you hide the DQ. Here is an example:

.. figure:: _static/img/N20170525S0116_stack_nomask.png
   :align: center

   Sky Subtracted and Stacked Final Image.

Note that the exposed image is clear but that the non illuminated region has
some cosmic rays leftovers that persisted even after the stack process.

.. todo @bquint The image above have some problems in the gaps. How do I fix
    them?
