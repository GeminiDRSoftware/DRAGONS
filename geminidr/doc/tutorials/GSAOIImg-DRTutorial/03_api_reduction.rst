.. 03_api_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb


.. _api_data_reduction:

Reduction using API
*******************

There may be cases where you might be interested in accessing the DRAGONS'
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. In this case, you will need to access
DRAGONS' tools by importing the appropriate modules and packages.


Importing Libraries
-------------------

Here are all the packages and modules that you will have to import for running
this tutorial::

    >>> import glob
    >>> import os
    >>>
    >>> from gempy.adlibrary import dataselect
    >>> from recipe_system import cal_service
    >>> from recipe_system.reduction.coreReduce import Reduce


The first two packages, :mod:`glob` and :mod:`os`, are Python built-in packages.
Here, :mod:`os` will be used to perform operations with the files names and
:mod:`glob` will be used to return a list with the input file names.

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
initialize it::

    >>> calibration_service = cal_service.CalibrationService()
    >>> calibration_service.config()
    >>> calibration_service.init(wipe=True)
    >>>
    >>> cal_service.set_calservice()

The ``wipe=True`` can be omitted if you want to keep old calibration files that
were added to the local database.


Create list of files
--------------------

Here, again, we have to create lists of files that will be used on each of the
data reduction step. We can start by creating a list will all the file names::

    >>> all_files = glob.glob(os.path.join(test_path, 'raw/'*.fits'))

Where ``test_path`` is a string containing the path where you saved your data.

Now we can use the ``all_files`` list as an input to
:func:`~gempy.adlibrary.dataselect.select_data`. Your will may have to add
a list of matching Tags, a list of excluding Tags and an expression that has
to be parsed by :func:`~gempy.adlibrary.dataselect.expr_parser`. These three
arguments are positional arguments (position matters) and they are separated
by comma.

As an example, let us can select the files that will be used to create a master
DARK frame. Remember that **GSAOI data does not require DARK correction**. So
this step is simply to make the tutorial complete::

    >>> darks_150s = dataselect.select_data(
    ...     all_files,
    ...     ['GSAOI', 'DARK', 'RAW'],
    ...     [],
    ...     dataselect.expr_parser('exposure_time==150')
    ... )
    ...

Note the empty list ``[]`` in the fourth line. It means that we are not passing
any argument for the Tags exclusion.

The lists with the FLAT images for ``Kshort`` and ``H`` using::

    >>> list_of_flats_Ks = dataselect.select_data(
    ...     all_files,
    ...     ['GSAOI', 'FLAT', 'RAW'],
    ...     [],
    ...     dataselect.expr_parser('filter_name=="Kshort"')
    ... )
    ...
    >>> list_of_flats_H = dataselect.select_data(
    ...     all_files,
    ...     ['GSAOI', 'FLAT', 'RAW'],
    ...     [],
    ...     dataselect.expr_parser(' filter_name=="H" ')
    ... )
    ...


For the standard start selection, we use::

    >>> list_of_std_stars = dataselect.select_data(
    ...     all_files,
    ...     [],
    ...     [],
    ...     dataselect.expr_parser('observation_class=="partnerCal"')
    ... )
    ...

Here, we are passing empty lists to the second and the third argument since
we do not need to use the Tags for selection nor for exclusion.

Finally, the science data can be selected using::

    >>> list_of_science_images = dataselect.select_data(
    ...     all_files,
    ...     [],
    ...     [],
    ...     dataselect.expr_parser('(observation_class=="science" and exposure_time==60.)')
    ... )
    ...


.. _api_process_dark_files:

Process DARK files
------------------

Again, accordingly to the `Calibration webpage for GSAOI
<https://www.gemini.edu/sciops/instruments/gsaoi/calibrations>`_,
**DARK subtraction is not necessary** since the dark noise level is too low.
DARK files are only used to generate Bad Pixel Masks (BPM).

If, for any reason, you believe that you really need to have a master DARK file,
you can create it using the commands below: ::

   >>> reduce_darks = Reduce()
   >>> reduce_darks.files.extend(darks_150s)
   >>> reduce_darks.runr()

The first line creates an instance of the
:class:`~recipe_system.reduction.coreReduce.Reduce` class. It is responsible to
check on the first image in the input list and find what is the appropriate
Recipe it should apply. The second line passes the list of dark frames to the
:class:`~recipe_system.reduction.coreReduce.Reduce` ``files`` attribute.
The :meth:`~recipe_system.reduction.coreReduce.Reduce.runr` triggers the
start of the data reduction.


.. _api_create_bpm_files:

Create BPM files
----------------


.. _api_process_flat_files:

Process FLAT files
------------------


.. _api_process_science_files:

Process Science files
---------------------


.. _api_stack_science_images:

Stack Science reduced images
----------------------------

