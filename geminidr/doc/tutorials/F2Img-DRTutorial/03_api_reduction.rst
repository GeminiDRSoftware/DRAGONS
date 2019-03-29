.. 03_api_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. |github| image:: /_static/img/GitHub-Mark-32px.png
    :scale: 75%


.. _api_data_reduction:

Reduction using API
*******************

There may be cases where you might be interested in accessing the DRAGONS'
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. In this scenario, you will need to access
DRAGONS' tools by importing the appropriate modules and packages.


Importing Libraries
-------------------

Here are all the packages and modules that you will have to import for running
this tutorial:

.. code-block:: python
    :linenos:

    import glob
    import os

    from gempy.adlibrary import dataselect
    from recipe_system import cal_service
    from recipe_system.reduction.coreReduce import Reduce


The first two packages, :mod:`glob` and :mod:`os`, are Python built-in packages.
Here, :mod:`os` will be used to perform operations with the files names and
:mod:`glob` will be used to return a :class:`list` with the input file names.

.. todo: check references

Then, we are importing the :mod:`~gempy.adlibrary.dataselect` from the
:mod:`gempy.adlibrary`. It will be used to select the data in the same way we
did as in :ref:`create_file_lists` section. The
:mod:`~recipe_system.cal_service` package will be our interface with the
local calibration database. Finally, the
:class:`~recipe_system.reduction.coreReduce.Reduce` class will be
used to actually run the data reduction pipeline.


The Calibration Service
-----------------------

Before we start, let's be sure we have properly setup our database. First
create the `rsys.cfg` file as described in
`the caldb documentation in the Recipe System User's Manual <caldb>`_. Then,
you can use the following commands to configure the local database and
initialize it:

.. code-block:: python
    :linenos:
    :lineno-start: 7

    calibration_service = cal_service.CalibrationService()
    calibration_service.config()
    calibration_service.init(wipe=True)

    cal_service.set_calservice()


The ``wipe=True`` can be omitted if you want to keep old calibration files that
were added to the local database.

Create :class:`list` of files
-----------------------------

Here, again, we have to create lists of files that will be used on each of the
data reduction step. We can start by creating a :class:`list` will all the file
names:

.. code-block:: python
    :linenos:
    :lineno-start: 12

    all_files = glob.glob('./raw/*.fits')

Where the string between parenthesis means that we are selecting every file that
ends with ``.fits`` and that lives withing the ``./raw`` directory. Before you
carry on, we recommend that you use ``print(all_files)`` to check if they were
properly read.

Now we can use the ``all_files`` :class:`list` as an input to
:func:`~gempy.adlibrary.dataselect.select_data`. Your will may have to add
a :class:`list` of matching Tags, a :class:`list` of excluding Tags and an expression that has
to be parsed by :func:`~gempy.adlibrary.dataselect.expr_parser`. These three
arguments are positional arguments (position matters) and they are separated
by comma.

As an example, let us can select the files that will be used to create a master
DARK frame for the files that have 20s exposure time:

.. code-block:: python
    :linenos:
    :lineno-start: 13

    dark_files_3s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==3')
    )

    dark_files_8s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==8')
    )

    dark_files_15s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==15')
    )

    dark_files_20s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==20')
    )

    dark_files_60s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==60')
    )

    dark_files_120s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==120')
    )

Note the empty list ``[]`` in the fourth line of each command. This
position argument receives a list of tags that will be used to exclude
any files with the matching tag from our selection.

Now you must create a list of FLAT images for each filter. You can do that by
using the following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 54

    list_of_flats_Y = dataselect.select_data(
         all_files,
         ['F2', 'FLAT', 'RAW'],
         [],
         dataselect.expr_parser('filter_name=="Y"')
    )

Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 60

    list_of_science_images = dataselect.select_data(
        all_files,
        ['F2'],
        [],
        dataselect.expr_parser('(observation_class=="science" and filter_name=="Y")')
    )


.. _api_process_dark_files:

Process DARK files
------------------

For each exposure time, we will have to run the command lines below:

.. code-block:: python
   :linenos:
   :lineno-start: 66

    reduce_darks = Reduce()
    reduce_darks.files.extend(dark_files_003s)
    reduce_darks.runr()

    calibration_service.add_cal(reduce_darks.output_filenames[0])

The first line creates an instance of the
:class:`~recipe_system.reduction.coreReduce.Reduce` class. It is responsible to
check on the first image in the input :class:`list` and find what is the
appropriate Recipe it should apply. The second line passes the :class:`list` of
dark frames to the :class:`~recipe_system.reduction.coreReduce.Reduce`
``files`` attribute. The
:meth:`~recipe_system.reduction.coreReduce.Reduce.runr` triggers the start of
the data reduction.

Instead of repeating the code block above, you can simply use a ``for`` loop:

.. code-block:: python
   :linenos:
   :lineno-start: 71

    for dark_list in [dark_files_3s, dark_files_8s, dark_files_15s,
                     dark_files_20s, dark_files_60s, dark_files_120s]:

        reduce_darks = Reduce()
        reduce_darks.files.extend(dark_list)
        reduce_darks.runr()

        calibration_service.add_cal(reduce_darks.output_filenames[0])

.. _api_create_bpm_files:

Create BPM files
----------------

The Bad Pixel Mask files can be easily created using the follow commands:

.. code-block:: python
    :linenos:
    :lineno-start: 79

    reduce_bpm = Reduce()
    reduce_bpm.files.extend(list_of_flats_Y)
    reduce_bpm.files.extend(dark_files_3s)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

Note that, here, we are setting the recipe name to 'makeProcessedBPM' on
line 82.


.. _api_process_flat_files:

Process FLAT files
------------------

We can now reduce our FLAT files by using the following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 84

    bpm_filename = reduce_bpm.output_filenames[0]

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats_Y)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    calibration_service.add_cal(reduce_flats.output_filenames[0])

On Line 84, we get the first (only) output file from the ``reduce_bpm`` pipeline
and store it in the ``bpm_filename`` variable. Then, we pass it to the
``reduce_flats`` pipeline by updating the ``.uparms`` attribute. Remember
that ``.uparms`` must be a :class:`list` of :class:`Tuples`.

After the pipeline, we add master flat file to the calibration manager using
the line 91.


.. _api_process_science_files:

Process Science files
---------------------

Finally, we can use similar commands to create a new pipeline and reduce the
science data:

.. code-block:: python
    :linenos:
    :lineno-start: 92

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_images)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.runr()


