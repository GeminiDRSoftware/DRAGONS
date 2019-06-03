.. 03_api_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. _caldb_api: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/caldb.html#using-the-caldb-api

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

.. todo @bquint: the gempy auto-api is not being generated anywhere.
.. todo:: @bquint the gempy auto-api is not being generated anywhere. Find a
    place for it.

Then, we are importing the :mod:`~gempy.adlibrary.dataselect` from the
:mod:`gempy.adlibrary`. It will be used to select the data in the same way we
did as in :ref:`create_file_lists` section. The
:mod:`~recipe_system.cal_service` package will be our interface with the
local calibration database. Finally, the
:class:`~recipe_system.reduction.coreReduce.Reduce` class will be
used to actually run the data reduction pipeline.

When using the API, you will notice that the output messages appear twice.
To prevent this behaviour you can set one of the output stream to a file
using the :mod:`gempy.utils.logutils` module and its
:func:`~gempy.utils.logutils.config()` function:


.. code-block:: python
    :linenos:
    :lineno-start: 7

    from gempy.utils import logutils
    logutils.config(file_name='dummy.log')


.. _set_caldb_api:

The Calibration Service
-----------------------

Before we start, let's be sure we have properly setup our database.

First, check that you have already a ``rsys.cfg`` file inside the
``~/.geminidr/``. It should contain:

.. code-block:: none

    [calibs]
    standalone = True
    database_dir = ${path_to_my_data}/f2img_tutorial/playground


This simply tells the system where to put the calibration database. This
database will keep track of the processed calibrations as we add these files
to it.

..  note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.geminidr``.

The calibration database is initialized and the calibration service is
configured like this:

.. code-block:: python
    :linenos:
    :lineno-start: 9

    calibration_service = cal_service.CalibrationService()
    calibration_service.config()
    calibration_service.init()

    cal_service.set_calservice()

The calibration service is now ready to use. If you need more details,
check the `Using the caldb API in the Recipe System User's Manual
<caldb_api>`_.


.. _create_file_lists:

Create :class:`list` of files
-----------------------------

Here, again, we have to create lists of files that will be used on each of the
data reduction step. We can start by creating a :class:`list` will all the file
names:

.. code-block:: python
    :linenos:
    :lineno-start: 14

    all_files = glob.glob('../playdata/*.fits')
    all_files.sort()

Where the string between parenthesis means that we are selecting every file that
ends with ``.fits`` and that lives withing the ``../playdata/`` directory.
The :meth:`~list.sort` method simply re-organize the list with the file names
and is an optional step. Before you carry on, we recommend that you use
``print(all_files)`` to check if they were properly read.

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
    :lineno-start: 16

    dark_files_20s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==20')
    )

Note the empty list ``[]`` in the fourth line of each command. This
position argument receives a list of tags that will be used to exclude
any files with the matching tag from our selection (i.e., equivalent to the
``--xtags`` option).

We can now repeat the same syntax for the darks with 3 and 120 seconds:

.. code-block:: python
    :linenos:
    :lineno-start: 22

    dark_files_3s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==3')
    )

    dark_files_120s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==120')
    )

Now you must create a list of FLAT images for each filter. You can do that by
using the following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 35

    list_of_flats_Y = dataselect.select_data(
         all_files,
         ['F2', 'FLAT', 'RAW'],
         [],
         dataselect.expr_parser('filter_name=="Y"')
    )

Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 41

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
   :lineno-start: 47

    reduce_darks = Reduce()
    reduce_darks.files.extend(dark_files_3s)
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
   :lineno-start: 52

    for dark_list in [dark_files_3s, dark_files_20s, dark_files_120s]:

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
    :lineno-start: 59

    reduce_bpm = Reduce()
    reduce_bpm.files.extend(list_of_flats_Y)
    reduce_bpm.files.extend(dark_files_3s)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

Note that, here, we are setting the recipe name to 'makeProcessedBPM' on
line 62.


.. _api_process_flat_files:

Process FLAT files
------------------

We can now reduce our FLAT files by using the following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 64

    bpm_filename = reduce_bpm.output_filenames[0]

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats_Y)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    calibration_service.add_cal(reduce_flats.output_filenames[0])

On Line 64, we get the first (only) output file from the ``reduce_bpm`` pipeline
and store it in the ``bpm_filename`` variable. Then, we pass it to the
``reduce_flats`` pipeline by updating the ``.uparms`` attribute. Remember
that ``.uparms`` must be a :class:`list` of :class:`Tuples`.

Once :meth:`runr()` is finished, we add master flat file to the calibration manager
using the line 71.


.. _api_process_science_files:

Process Science files
---------------------

Finally, we can use similar commands to create a new pipeline and reduce the
science data:

.. code-block:: python
    :linenos:
    :lineno-start: 72

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_images)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.runr()


