.. ex1_gmosim_starfield_api.rst

.. _starfield_api:

**************************************************************
Example 1 - Star field with dithers - Using the "Reduce" class
**************************************************************

A reduction can be initiated from the command line as shown in
:ref:`starfield_cmdline` and it can also be done programmatically as we will
show here.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this example we replicate the
command line version of Example 1 but using the Python
programmatic interface. What is shown here could be packaged in modules for
greater automation.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`starfield_dataset`.

Here is a copy of the table for quick reference.

+---------------+---------------------+--------------------------------+
| Science       || N20170614S0201-205 || 10 s, i-band                  |
+---------------+---------------------+--------------------------------+
| Bias          || N20170613S0180-184 |                                |
|               || N20170615S0534-538 |                                |
+---------------+---------------------+--------------------------------+
| Twilight Flats|| N20170702S0178-182 || 40 to 16 s, i-band            |
+---------------+---------------------+--------------------------------+
| BPM           || bpm_20170306_gmos-n_Ham_22_full_12amp.fits          |
+---------------+------------------------------------------------------+

Setting Up
==========

Importing Libraries
-------------------

We first import the necessary modules and classes:

.. code-block:: python
    :linenos:

    import glob

    import astrodata
    import gemini_instruments
    from recipe_system.reduction.coreReduce import Reduce
    from gempy.adlibrary import dataselect


The ``dataselect`` module will be used to create file lists for the
biases, the flats, the arcs, the standard, and the science observations.
The ``Reduce`` class is used to set up and run the data
reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger. (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 7

    from gempy.utils import logutils
    logutils.config(file_name='gmos_data_reduction.log')


.. _set_caldb_api:

Setting up the Calibration Service
----------------------------------

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_api`.


.. _api_create_file_lists:

Create list of files
====================

The next step is to create input file lists. The module ``dataselect`` helps
with that.  It uses Astrodata tags and |descriptors| to select the files and
store the filenames to a Python list that can then be fed to the ``Reduce``
class. (See the |astrodatauser| for information about Astrodata and for a list
of |descriptors|.)

The first list we create is a list of all the files in the ``playdata/example1/``
directory.

.. code-block:: python
    :linenos:
    :lineno-start: 9

    all_files = glob.glob('../playdata/example1/*.fits')
    all_files.sort()

The :meth:`~list.sort` method simply re-organize the list with the file names
and is an optional step, but a recommended step. Before you carry on, you might want to do
``print(all_files)`` to check if they were properly read.

We will search that list for files with specific characteristics.  We use
the ``all_files`` :class:`list` as an input to the function
``dataselect.select_data()`` .  The function's signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')

We show several usage examples below.

List of Biases
--------------

Let us select the files that will be used to create a master bias:

.. code-block:: python
    :linenos:
    :lineno-start: 11

    list_of_biases = dataselect.select_data(
        all_files,
        ['BIAS'],
        []
    )

Note the empty list ``[]`` in line 20. This positional argument receives a list
of tags that will be used to exclude any files with the matching tag from our
selection (i.e., equivalent to the ``--xtags`` option).


List of Flats
-------------

Next we create a list of twilight flats for each filter. The expression
specifying the filter name is needed only if you have data from multiple
filters. It is not really needed in this case.

.. code-block:: python
    :linenos:
    :lineno-start: 16

    list_of_flats = dataselect.select_data(
        all_files,
        ['FLAT'],
        [],
        dataselect.expr_parser('filter_name=="i"')
    )

.. note::  All expressions need to be processed with ``dataselect.expr_parser``.


List of Science Data
--------------------

Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 22

    list_of_science = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('(observation_class=="science" and filter_name=="i")')
    )

Here we left the ``tags`` argument as an empty list and passed the tag
``'CAL'`` as an exclusion tag through the ``xtags`` argument.

We also added a fourth argument which is not necessary for our current dataset
but that can be useful for others. It contains an expression that has to be
parsed by ``dataselect.expr_parser``, and which ensures
that we are getting *science* frames obtained with the *i-band* filter.

Bad Pixel Mask
==============
Starting with DRAGONS v3.1, the static bad pixel masks (BPMs) are now handled
as calibrations.  They
are downloadable from the archive instead of being packaged with the software.
They are automatically associated like any other calibrations.  This means that
the user now must download the BPMs along with the other calibrations and add
the BPMs to the local calibration manager.

See :ref:`getBPM` in :ref:`tips_and_tricks` to learn about the various ways
to get the BPMs from the archive.

To add the BPM included in the data package to the local calibration database:

.. code-block:: python
    :linenos:
    :lineno-start: 28

    for bpm in dataselect.select_data(all_files, ['BPM']):
        caldb.add_cal(bpm)


.. _api_process_bias_files:

Make Master Bias
================

We create the master bias and add it to the calibration manager as follows:

.. code-block:: python
   :linenos:
   :lineno-start: 30

   reduce_bias = Reduce()
   reduce_bias.files.extend(list_of_biases)
   reduce_bias.runr()

The ``Reduce`` class is our reduction
"controller". This is where we collect all the information necessary for
the reduction. In this case, the only information necessary is the list of
input files which we add to the ``files`` attribute. The
``Reduce.runr`` method is where the
recipe search is triggered and where it is executed.

.. note:: The file name of the output processed bias is the file name of the
    first file in the list with ``_bias`` appended as a suffix.  This is the
    general naming scheme used by the ``Recipe System``.

.. note:: If you wish to inspect the processed calibrations before adding them
    to the calibration database, remove the "store" option attached to the
    database in the ``dragonsrc`` configuration file.  You will then have to
    add the calibrations manually following your inspection, eg.

    .. code-block::

       caldb.add_cal(reduce_bias.output_filenames[0])

.. _api_process_flat_files:

Make Master Flat
================

We create the master flat field and add it to the calibration database as follows:

.. code-block:: python
    :linenos:
    :lineno-start: 33

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats)
    reduce_flats.runr()


.. _api_process_fringe_frame:

Make Master Fringe Frame
========================

.. warning:: The dataset used in this tutorial does not require fringe
    correction so we skip this step.  To find out how to produce a master
    fringe frame, see :ref:`process_fringe_frame` in the
    :ref:`tips_and_tricks` chapter.


.. _api_process_science_files:

Reduce Science Images
=====================

We use similar statements as before to initiate a new reduction to reduce the
science data:

.. code-block:: python
    :linenos:
    :lineno-start: 36

    reduce_science = Reduce()
    reduce_science.files.extend(list_of_science)
    reduce_science.runr()

The output stack units are in electrons (header keyword BUNIT=electrons).
The output stack is stored in a multi-extension FITS (MEF) file.  The science
signal is in the "SCI" extension, the variance is in the "VAR" extension, and
the data quality plane (mask) is in the "DQ" extension.


Below are one of the raw images and the final stack:

.. figure:: _static/img/N20170614S0201.png
   :align: center

   One of the multi-extensions files.


.. figure:: _static/img/N20170614S0201_stack.png
   :align: center

   Final stacked image. The light-gray area represents the
   masked pixels.

