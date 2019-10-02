.. 02_data_reduction.rst

.. include:: DRAGONSlinks.txt

.. _data quality plane: https://astrodata-user-manual.readthedocs.io/en/latest/data.html#data-quality-plane

.. _command_line_data_reduction:

**************
Data Reduction
**************

This chapter will guide you on reducing **GMOS imaging data** using
command line tools. In this example we reduce a GMOS observation star field.
The observation is a simple dither-on-target sequence.
Just open a terminal to get started.

While the example cannot possibly cover all situations, it will help you get
acquainted with the reduction of GMOS data with DRAGONS. We encourage you to
look at the :ref:`tips_and_tricks` and :ref:`issues_and_limitations` chapters to
learn more about GMOS data reduction.

DRAGONS installation comes with a set of handful scripts that are used to
reduce astronomical data. The most important script is called
"|reduce|", which is extensively explained in the `Recipe System Users Manual
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/index.html>`_.
It is through that command that a DRAGONS reduction is launched.

For this tutorial, we will be also using other `Supplemental tools
<https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html>`_,
like:

* "|dataselect|"
* "|showd|"
* "|typewalk|"
* "|caldb|"


The dataset
===========

If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`about_data_set`.

Here is a copy of the table for quick reference.

+---------------+---------------------+--------------------------------+
| Science       || N20170614S0201-205 || 10 s, i-band                  |
+---------------+---------------------+--------------------------------+
| Bias          || N20170613S0180-184 |                                |
|               || N20170615S0534-538 |                                |
+---------------+---------------------+--------------------------------+
| Twilight Flats|| N20170702S0178-182 || 40 to 16 s, i-band            |
+---------------+---------------------+--------------------------------+



.. _setup_caldb:

Set up the Local Calibration Manager
====================================

DRAGONS comes with a local calibration manager that uses the same calibration
association rules as the Gemini Observatory Archive. This allows ``reduce``
to make requests to a local ligth-weight database for matching **processed**
calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, create or edit the configuration file ``rsys.cfg`` as
follow:

.. code-block:: none

    [calibs]
    standalone = True
    database_dir = /path_to_my_data/gmosimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

..  note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.geminidr``.

Then initialize the calibration database:

.. code-block:: bash

    caldb init

That's it! It is ready to use!  You can check the configuration and confirm
the setting with ``caldb config``.

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file **only** from the database
(it will **not** remove the file on disk). For more the details, check the
"|caldb|" documentation in the
`Recipe System: User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/>`_.

.. note:: If you have problems setting up "|caldb|" or want to bypass it for
      another reason, you can check the
      :ref:`Bypassing automatic calibration association <bypassing_caldb>`
      section.


.. _check_files:

Check files
===========

For this example, all the raw files we need are in the same directory called
``../playdata/``. Let us learn a bit about the data we have.

Ensure that you are in the ``playground`` directory and that the ``conda``
environment that includes DRAGONS has been activated.

Let us call the command tool "|typewalk|":

..  code-block:: bash

    $ typewalk -d ../playdata/

    directory:  /data/workspace/gmosimg_tutorial/playdata
     N20170613S0180.fits ............... (AT_ZENITH) (AZEL_TARGET) (BIAS) (CAL) (GEMINI) (GMOS) (NON_SIDEREAL) (NORTH) (RAW) (UNPREPARED)
     ...
     N20170614S0201.fits ............... (GEMINI) (GMOS) (IMAGE) (NORTH) (RAW) (SIDEREAL) (UNPREPARED)
     ...
     N20170615S0534.fits ............... (AT_ZENITH) (AZEL_TARGET) (BIAS) (CAL) (GEMINI) (GMOS) (NON_SIDEREAL) (NORTH) (RAW) (UNPREPARED)
     ...
     N20170702S0182.fits ............... (CAL) (FLAT) (GEMINI) (GMOS) (IMAGE) (NORTH) (RAW) (SIDEREAL) (TWILIGHT) (UNPREPARED)
    Done DataSpider.typewalk(..)


This command will open every FITS file within the folder passed after the ``-d``
flag (recursively) and will print an unsorted table with the file names and the
associated tags. For example, calibration files will always have the ``CAL``
tag. Flat images will always have the ``FLAT`` tag. This means that we can start
getting to know a bit more about our data set just by looking the tags. The
output above was trimmed for presentation.


.. _create_file_lists:

Create File lists
=================

This data set contains science and calibration frames. For some programs, it
could have different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you. You
have to do it. DRAGONS provides tools to help you with that.

The first step is to create lists that will be used in the data reduction
process. For that, we use "|dataselect|". Please, refer to the "|dataselect|"
documentation for details regarding its usage.

List of Biases
--------------

The bias files are selected with "|dataselect|":

..  code-block:: bash

    $ dataselect --tags BIAS ../playdata/*.fits -o list_of_bias.txt

List of Flats
-------------

Now we can do the same with the FLAT files:

..  code-block:: bash

    $ dataselect --tags FLAT ../playdata/*.fits -o list_of_flats.txt


If your dataset has flats obtained with more than one filter, you can add the
``--expr 'filter_name=="i"'`` expression to get on the flats obtained with in
the i-band. For example:

.. code-block:: bash

    $ dataselect --tags FLAT --expr 'filter_name=="i"' ../playdata/*.fits -o list_of_flats.txt


List for science data
---------------------

The rest is the data with your science target. The simplest way, in this case,
of creating a list of science frames is excluding everything that is a
calibration:

.. code-block:: bash

    $ dataselect --xtags CAL ../playdata/*.fits -o list_of_science.txt


This will work for our dataset because we know that a single target was observed
with a single filter and with the same exposure time. But what if we don't know
that?

We can check it by passing the "|dataselect|" output to the "|showd|" command
line using a "pipe" (``|``):

..  code-block:: bash

    $ dataselect --expr 'observation_class=="science"' ../playdata/*.fits | showd -d object,exposure_time
    -----------------------------------------------------------
    filename                             object   exposure_time
    -----------------------------------------------------------
    ../playdata/N20170614S0201.fits   starfield            10.0
    ../playdata/N20170614S0202.fits   starfield            10.0
    ../playdata/N20170614S0203.fits   starfield            10.0
    ../playdata/N20170614S0204.fits   starfield            10.0
    ../playdata/N20170614S0205.fits   starfield            10.0


The ``-d`` flag tells "|showd|" which "|descriptors|" will be printed for
each input file. As you can see, we have only observed target and only
exposure time.

To select on target name and exposure time, specify the criteria in the
``expr`` field of "|dataselect|":

.. code-block:: bash

   $ dataselect --expr '(object=="starfield" and exposure_time==10.)' ../playdata/*.fits -o list_of_science.txt


We have our input lists and we have initialized the calibration database, we
are ready to reduce the data.

Please make sure that you are still in the ``playground`` directory.


.. _make_master_bias:

Create a Master Bias
====================

We start the data reduction by creating a master bias for the science data.
It can be created and added to the calibration database using the commands below:

..  code-block:: bash

   $ reduce @list_of_bias.txt
   $ caldb add N20170613S0180_bias.fits


The ``@`` character before the name of the input file is the "at-file" syntax.
More details can be found in the |atfile| documentation.

To check that the master bias was added to the database, use ``caldb list``.


.. note::
    The master bias will be saved in the same folder where "|reduce|" was
    called *and* inside the ``./calibrations/processed_bias`` folder. The latter
    location is to cache a copy of the file. This applies to all the processed
    calibration.

    Some people might prefer adding the copy in the ``calibrations`` directory
    as it is safe from a ``rm *``, for example.

    .. code-block:: bash

        $ caldb add ./calibrations/processed_dark/N20170613S0180_bias.fits

.. note::
    "|reduce|" uses the first filename in the input list as basename and adds
    ``_bias`` as a suffix to it. So if your first filename is, for example,
    ``N20170613S0180.fits``, the output will be `N20170613S0180_bias.fits``.


.. _process_flat_files:

Create a Master Flat Field
==========================

Twilight flats images are used to produced an imaging master flat and the
result is added to the calibration database.

..  code-block:: bash

   $ reduce @list_of_flats.txt
   $ caldb add N20170702S0178_flat.fits


Note "|reduce|" will query the local calibration manager for the master bias
and use it in the data reduction.

Once finished you will have the master flat in the current work directory and
inside ``./calibrations/processed_flat``. It will have a ``_flat`` suffix.


Create Master Fringe Frame
==========================

.. warning:: The dataset used in this tutorial does not require fringe
    correction so we skip this step.  To find out how to produce a master
    fringe frame, see :ref:`process_fringe_frame` in the
    :ref:`tips_and_tricks` chapter.


.. _processing_science_files:

Reduce Science Images
=====================

Once we have our calibration files processed and added to the database, we can
run ``reduce`` on our science data:

.. code-block:: bash

   $ reduce @list_of_science.txt

This command will generate bias and flat corrected files and will stack them.
If a fringe frames is needed this command will apply the correction.  The stacked
image will have the ``_stack`` suffix.

The output stack units are in electrons (header keyword BUNIT=electrons).
The output stack is stored in a multi-extension FITS (MEF) file.  The science
signal is in the "SCI" extension, the variance is in the "VAR" extension, and
the data quality plane (mask) is in the "DQ" extension.


.. note::  Depending on your version of Astropy, you might see a lot of
    Astropy warnings about headers and coordinates system.  You can safely
    ignore them.

Below are one of the raw images and the final stack:

.. figure:: _static/img/N20170614S0201.png
   :align: center

   One of the multi-extensions files.


.. figure:: _static/img/N20170614S0201_stack.png
   :align: center

   Final stacked image. The light-gray area represents the
   masked pixels.

