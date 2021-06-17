.. 03_multisrc_cmdline.rst

.. include:: DRAGONSlinks.txt

.. _multisrc_cmdline:

********************************************************************
Example 1-A: Multi-source Longslit - Using the "reduce" command line
********************************************************************

In this example we will reduce a GMOS longslit observation of multiple stars
using the "|reduce|" command that is operated directly from the unix shell.
Just open a terminal and load the DRAGONS conda environment to get started.

This observation dithers along the slit and along the dispersion axis.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`_datamultisrc`.

Here is a copy of the table for quick reference.

+---------------------+---------------------------------+
| Science             || S20180419S0041-42 (650 nm)     |
|                     || S20180419S0045-46 (660 nm)     |
+---------------------+---------------------------------+
| Science biases      || S20180419S0236-240             |
|                     || S20180420S0222-226             |
+---------------------+---------------------------------+
| Science flats       || S20180419S0040 (650 nm)        |
|                     || S20180419S0043 (650 nm)        |
|                     || S20180419S0044 (660 nm)        |
|                     || S20180419S0047 (660 nm)        |
+---------------------+---------------------------------+
| Science arcs        || S20180420S0019 (650 nm)        |
|                     || S20180420S0020 (660 nm)        |
+---------------------+---------------------------------+
| Standard (EG131)    || S20180420S0200 (650 nm)        |
+---------------------+---------------------------------+
| Standard biases     || S20180419S0236-240             |
|                     || S20180420S0222-226             |
+---------------------+---------------------------------+
| Standard flats      || S20180420S0201 (650nm)         |
+---------------------+---------------------------------+
| Standard arc        || S20180420S0301 (650 nm)        |
+---------------------+---------------------------------+


Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and a local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows "|reduce|" to make requests for matching **processed**
calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, create or edit the configuration file ``rsys.cfg`` as
follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/gmosls_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibrations we are going to
send to it.

.. note:: ``~`` in the path above refers to your home directory.  Also, don't
    miss the dot in ``.geminidr``.

Then initialize the calibration database::

    caldb init

That's it.  It is ready to use.

You can add processed calibrations with ``caldb add <filename>`` (we will
later), list the database content with ``caldb list``, and
``caldb remove <filename>`` to remove a file from the database (it will **not**
remove the file on disk.)  (See the "|caldb|" documentation for more details.)


Create file lists
=================

This data set contains science and calibration frames. For some programs, it
could have different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata.)

First, navigate to the ``playground`` directory in the unpacked data package.


A list for the biases
---------------------
The science observations and the spectrophotometric standard observations were
obtained just one day apart.  We will use the same master bias for both.

::

    dataselect ../playdata/*.fits