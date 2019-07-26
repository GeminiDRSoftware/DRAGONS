.. 03_api_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. _primitive: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/definitions.html#primitive


.. |github| image:: /_static/img/GitHub-Mark-32px.png
    :scale: 75%


.. _api_data_reduction:

Reduction using API
*******************

There may be cases where you might be interested in accessing the DRAGONS'
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. In this scenario, you will need to access
DRAGONS' tools by importing the appropriate modules and packages.

Again, remember that our working directory will be
``/path_to_my_data/playground/`` .


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
    logutils.config(file_name='gmos_data_reduction.log')


.. _set_caldb_api:

The Calibration Service
-----------------------

Before we start, let's be sure we have properly setup our database.

First, check that you have already a ``rsys.cfg`` file inside the
``~/.geminidr/``. It should contain:

.. code-block:: none

    [calibs]
    standalone = True
    database_dir = /path_to_my_data/gmosimg_tutorial_api/playground


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
check the
`Using the caldb API in the Recipe System User's Manual <https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/caldb.html#using-the-caldb-api>`_ .

..  todo: calmanager
..  warning:: The Gemini Local Calibration Manager is not available yet in the
    Gemini Conda Channel for installation and you might not have it installed.
    If you get an error with the message
    `NameError: name 'localmanager' is not defined` when running line 10, you
    don't the Local Calibration Manager installed. For now, please, contact
    someone in the Gemini Science User Support Department for more details.


.. _api_create_file_lists:

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
:func:`~gempy.adlibrary.dataselect.select_data`. Your will have to add a
:class:`list` of matching Tags and a :class:`list` of excluding Tags. These
three arguments are positional arguments (position matters) and they are
separated by comma.

As an example, let us can select the files that will be used to create a master
Bias frame:

.. code-block:: python
    :linenos:
    :lineno-start: 16

    list_of_biases = dataselect.select_data(
        all_files,
        ['BIAS'],
        []
    )

Note the empty list ``[]`` in line 19. This positional argument receives a list
of tags that will be used to exclude any files with the matching tag from our
selection (i.e., equivalent to the ``--xtags`` option).

Now you must create a list of FLAT images. You can do that by using the
following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 21

    list_of_flats = dataselect.select_data(
         all_files,
         ['FLAT'],
         []
    )

Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 26

    list_of_science = dataselect.select_data(
        all_files,
        [],
        ['CAL'],
        dataselect.expr_parser('(observation_class=="science" and filter_name=="g")')
    )

Here we left the ``TAGS`` argument as an empty list and passed the ``'CAL'`` as
an ``XTAGS`` argument.

We also added a fourth argument which is not necessary for our current dataset
but that can be useful for others. It contains an expression that has to be
parsed by :func:`~gempy.adlibrary.dataselect.expr_parser`, and which ensures
that we are getting science frames obtained with the g-band filter.


.. _api_process_bias_files:

Process Bias files
------------------

The Bias data reduction can be performed using the following commands:

.. code-block:: python
   :linenos:
   :lineno-start: 32

    reduce_bias = Reduce()
    reduce_bias.files.extend(list_of_biases)
    reduce_bias.runr()

    calibration_service.add_cal(reduce_bias.output_filenames[0])

The first line creates an instance of the
:class:`~recipe_system.reduction.coreReduce.Reduce` class. It is responsible to
check on the first image in the input :class:`list` and find what is the
appropriate Recipe it should apply. The second line passes the :class:`list` of
dark frames to the :class:`~recipe_system.reduction.coreReduce.Reduce`
``files`` attribute. The
:meth:`~recipe_system.reduction.coreReduce.Reduce.runr` method triggers the
start of the data reduction.

:meth:`~recipe_system.reduction.coreReduce.Reduce.runr` uses the first filename
in the input list as basename. So if your first filename is, for example,
``N20001231S001.fits``, the output will be ``N20001231S001_bias.fits``. Because
of that, the base name of the Master Bias file can be different for you.

.. _api_process_flat_files:

Process FLAT files
------------------

We can now reduce our FLAT files by using the following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 37

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats)
    reduce_flats.runr()

    calibration_service.add_cal(reduce_flats.output_filenames[0])


The code above is equivalent to what we did in the `api_process_bias_files`_: line
37 creates an instance of the :class:`~recipe_system.reduction.coreReduce.Reduce`
class, line 38 passes the lists with the flat files to the `reduce_flats.files`
attribute using the :meth:`~recipe_system.reduction.coreReduce.Reduce.files.extend`
method and line 39 starts the data reduction.

Once :meth:`~recipe_system.reduction.coreReduce.Reduce.runr` is finished, we add
master flat file to the calibration manager using the line 41. Here,
:meth:`~recipe_system.reduction.coreReduce.Reduce.runr` will create a file with
the ``_flat`` suffix.


.. _api_process_fring_frame:

Process Fringe Frame
--------------------

.. note:: The dataset used in this tutorial does not require Fringe Correction
    so you can skip this section if you are following it. Find more information
    below.

The reduction of some datasets requires a Processed Fringe Frame. The datasets
that need a Fringe Frame are shown in the appendix
`Fringe Correction Tables <fringe_correction_tables>`_.

If you find out that your dataset needs Fringe Correction, you can use the
code block below to create the Processed Fringe Frame:

.. code-block:: python
    :linenos:
    :lineno-start: 42

    reduce_fringe = Reduce()
    reduce_fringe.files.extend(list_of_science)
    reduce_fringe.runr()

    calibration_service.add_cal(reduce_fringe.output_filenames[0])

The code above is very similar to the code used for Bias and Flats. The output
file will have the ``_fringe`` suffix.


.. _api_process_science_files:

Process Science files
---------------------

Finally, we can use similar commands to create a new pipeline and reduce the
science data:

.. code-block:: python
    :linenos:
    :lineno-start: 47

    reduce_science = Reduce()
    reduce_science.files.extend(list_of_science)
    reduce_science.runr()

..  warning:: This is a heavy process computational speaking given the stack
    primitive_. Our team is working on this for better performance.

Again, if you need to change the parameters used in a given primitive_,
you can change its parameters. This can be done by appending parameters to
the :meth:`~recipe_system.reduction.coreReduce.Reduce.uparms` using the command
below:

.. code-block:: python
    :linenos:
    :lineno-start: 50

    reduce_science.uparms.append(("stackFrames:scale", True))

Before you run the pipeline again, you might want to change the suffix of the
output file. You can do that with:

.. code-block:: python
    :linenos:
    :lineno-start: 51

    reduce_science.suffix = "_scale_stack"
    reduce_science.runr()

..  warning:: Some primitives use a lot of computer memory and might freeze your
    computer. Make sure you save all your work before running
    :meth:`~recipe_system.reduction.coreReduce.Reduce.runr`.

