.. 03_api_reduction.rst

.. include:: DRAGONSlinks.txt

.. _primitive: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/definitions.html#primitive


.. |github| image:: /_static/img/GitHub-Mark-32px.png
    :scale: 75%


.. _api_data_reduction:

*******************
Reduction using API
*******************

There may be cases where you would be interested in accessing the DRAGONS'
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. Here we show you how to do the same
reduction we did in the previous chapter but using the API.


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

Setting Up
==========

Importing Libraries
-------------------

We first import the necessary modules and classes:

.. code-block:: python
    :linenos:

    from __future__ import print_function

    import glob

    from gempy.adlibrary import dataselect
    from recipe_system import cal_service
    from recipe_system.reduction.coreReduce import Reduce


Importing ``print_function`` is for compatibility with the Python 2.7 print
statement. If you are working with Python 3, it is not needed, but importing
it will not break anything.

:mod:`glob` is Python built-in packages. It will be used to return a
:class:`list` with the input file names.

.. todo @bquint: the gempy auto-api is not being generated anywhere. Find a
    place for it.


:mod:`~gempy.adlibrary.dataselect` will be used to create file lists for the
darks, the flats and the science observations. The
:mod:`~recipe_system.cal_service` package is our interface with the local
calibration database. Finally, the
:class:`~recipe_system.reduction.coreReduce.Reduce` class is used to set up
and run the data reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger. (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 8

    from gempy.utils import logutils
    logutils.config(file_name='gmos_data_reduction.log')


.. _set_caldb_api:

Setting up the Calibration Service
----------------------------------

Before we continue, let's be sure we have properly setup our calibration
database and the calibration association service.

First, check that you have already a ``rsys.cfg`` file inside the
``~/.geminidr/``. It should contain:

.. code-block:: none

    [calibs]
    standalone = True
    database_dir = /path_to_my_data/gmosimg_tutorial_api/playground


This tells the system where to put the calibration database. This
database will keep track of the processed calibrations as we add them
to it.

.. note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.geminidr``.

The calibration database is initialized and the calibration service is
configured as follow:

.. code-block:: python
    :linenos:
    :lineno-start: 10

    caldb = cal_service.CalibrationService()
    caldb.config()
    caldb.init()

    cal_service.set_calservice()

The calibration service is now ready to use. If you need more details,
check the
`Using the caldb API in the Recipe System User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/caldb.html#using-the-caldb-api>`_ .


.. _api_create_file_lists:

Create list of files
====================

The next step is to create lists of files that will be used as input to each of the
data reduction steps. Let us start by creating a :class:`list` of all the
FITS files in the directory ``../playdata/``.

.. code-block:: python
    :linenos:
    :lineno-start: 15

    all_files = glob.glob('../playdata/*.fits')
    all_files.sort()

The :meth:`~list.sort` method simply re-organize the list with the file names
and is an optional step. Before you carry on, you might want to do
``print(all_files)`` to check if they were properly read.

Now we can use the ``all_files`` :class:`list` as an input to
:func:`~gempy.adlibrary.dataselect.select_data`.  The
``dataselect.select_data()`` function signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')


List of Biases
--------------

Let us, now, select the files that will be used to create a master bias:

.. code-block:: python
    :linenos:
    :lineno-start: 17

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
    :lineno-start: 22

    list_of_flats = dataselect.select_data(
         all_files,
         ['FLAT'],
         [],
         dataselect.expr_parser('filter_name=="i"')
    )

List of Science Data
--------------------

Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 27

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
parsed by :func:`~gempy.adlibrary.dataselect.expr_parser`, and which ensures
that we are getting *science* frames obtained with the *i-band* filter.


.. _api_process_bias_files:

Make Master Bias
================

We create the master bias and add it to the calibration manager as follow:

.. code-block:: python
   :linenos:
   :lineno-start: 33

   reduce_bias = Reduce()
   reduce_bias.files.extend(list_of_biases)
   reduce_bias.runr()

   caldb.add_cal(reduce_bias.output_filenames[0])

The :class:`~recipe_system.reduction.coreReduce.Reduce` class is our reduction
"controller". This is where we collect all the information necessary for
the reduction. In this case, the only information necessary is the list of
input files which we add to the ``files`` attribute. The
:meth:`~recipe_system.reduction.coreReduce.Reduce.runr` method is where the
recipe search is triggered and where it is executed.

Once :meth:`runr()` is finished, we add the master bias to the calibration
manager (line 37).


.. _api_process_flat_files:

Make Master Flat
================

We create the master flat field and add it to the calibration database as follow:

.. code-block:: python
    :linenos:
    :lineno-start: 38

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats)
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])


.. _api_process_fring_frame:

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
    :lineno-start: 43

    reduce_science = Reduce()
    reduce_science.files.extend(list_of_science)
    reduce_science.runr()

The output stack units are in electrons (header keyword BUNIT=electrons).
The output stack is stored in a multi-extension FITS (MEF) file.  The science
signal is in the "SCI" extension, the variance is in the "VAR" extension, and
the data quality plane (mask) is in the "DQ" extension.
