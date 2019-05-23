.. 03_api_reduction.rst

.. _caldb: https://dragons-recipe-system-users-manual.readthedocs.io/en/latest/supptools.html#caldb

.. |github| image:: /_static/img/GitHub-Mark-32px.png
    :scale: 75%


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
data reduction step. We can start by creating a :class:`list` will all the file names:

.. code-block:: python
    :linenos:
    :lineno-start: 12

    all_files = glob.glob('../playdata/*.fits')

Where the string between parenthesis means that we are selecting every file that
ends with ``.fits`` and that lives withing the ``../playdata/`` directory. Before
you carry on, we recommend that you use ``print(all_files)`` to check if they
were properly read.

Now we can use the ``all_files`` :class:`list` as an input to
:func:`~gempy.adlibrary.dataselect.select_data`. Your will may have to add
a :class:`list` of matching Tags, a :class:`list` of excluding Tags and an expression that has
to be parsed by :func:`~gempy.adlibrary.dataselect.expr_parser`. These three
arguments are positional arguments (position matters) and they are separated
by comma.

As an example, let us can select the files that will be used to create a master
DARK frame. Remember that **GSAOI data does not require DARK correction**. So
this step is simply to make the tutorial complete:

.. code-block:: python
    :linenos:
    :lineno-start: 13

    darks_150s = dataselect.select_data(
        all_files,
        ['GSAOI', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==150')
    )
    

Note the empty :class:`list` ``[]`` in the fourth line. It means that we are not passing
any argument for the Tags exclusion.

The lists with the FLAT images for ``Kshort`` and ``H`` using:

.. code-block:: python
    :linenos:
    :lineno-start: 19

    list_of_flats_Ks = dataselect.select_data(
         all_files,
         ['GSAOI', 'FLAT', 'RAW'],
         [],
         dataselect.expr_parser('filter_name=="Kshort"')
    )

    list_of_flats_H = dataselect.select_data(
        all_files,
        ['GSAOI', 'FLAT', 'RAW'],
        [],
        dataselect.expr_parser(' filter_name=="H" ')
    )


For the standard start selection, we use:

.. code-block:: python
    :linenos:
    :lineno-start: 32

    list_of_std_stars = dataselect.select_data(
        all_files,
        [],
        [],
        dataselect.expr_parser('observation_class=="partnerCal"')
    )


Here, we are passing empty lists to the second and the third argument since
we do not need to use the Tags for selection nor for exclusion.

Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 38

    list_of_science_images = dataselect.select_data(
        all_files,
        [],
        [],
        dataselect.expr_parser('(observation_class=="science" and exposure_time==60.)')
    )


.. _api_process_dark_files:

Process DARK files
------------------

Again, accordingly to the `Calibration webpage for GSAOI
<https://www.gemini.edu/sciops/instruments/gsaoi/calibrations>`_,
**DARK subtraction is not necessary** since the dark noise level is too low.
DARK files are only used to generate Bad Pixel Masks (BPM).

If, for any reason, you believe that you really need to have a master DARK file,
you can create it using the commands below:

.. code-block:: python
   :linenos:
   :lineno-start: 44

    reduce_darks = Reduce()
    reduce_darks.files.extend(darks_150s)
    reduce_darks.runr()

The first line creates an instance of the
:class:`~recipe_system.reduction.coreReduce.Reduce` class. It is responsible to
check on the first image in the input :class:`list` and find what is the appropriate
Recipe it should apply. The second line passes the :class:`list` of dark frames to the
:class:`~recipe_system.reduction.coreReduce.Reduce` ``files`` attribute.
The :meth:`~recipe_system.reduction.coreReduce.Reduce.runr` triggers the
start of the data reduction.


.. _api_create_bpm_files:

Create BPM files
----------------

The Bad Pixel Mask files can be easily created using the follow commands:

.. code-block:: python
    :linenos:
    :lineno-start: 47

    reduce_bpm = Reduce()
    reduce_bpm.files.extend(list_of_flats_H)
    reduce_bpm.files.extend(darks_150s)
    reduce_bpm.recipename = 'makeProcessedBPM'
    reduce_bpm.runr()

Note that, here, we are setting the recipe name to 'makeProcessedBPM' on
line 50.


.. _api_process_flat_files:

Process FLAT files
------------------

We can now reduce our FLAT files by using the following commands:

.. code-block:: python
    :linenos:
    :lineno-start: 52

    bpm_filename = reduce_bpm.output_filenames[0]

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats_Ks)
    reduce_flats.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_flats.runr()

    calibration_service.add_cal(reduce_flats.output_filenames[0])

On Line 52, we get the first (only) output file from the ``reduce_bpm`` pipeline
and store it in the ``bpm_filename`` variable. Then, we pass it to the
``reduce_flats`` pipeline by updating the ``.uparms`` attribute. Remember
that ``.uparms`` must be a :class:`list` of :class:`Tuples`.

After the pipeline, we add master flat file to the calibration manager using
the line 59.


.. _api_process_science_files:

Process Science files
---------------------

We can use similar commands to create a new pipeline and reduce the science
data:

.. code-block:: python
    :linenos:
    :lineno-start: 60

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_images)
    reduce_target.uparms = [('addDQ:user_bpm', bpm_filename)]
    reduce_target.runr()


.. _api_stack_science_images:

Stack Science reduced images
----------------------------

.. todo::

  ?BQ? make .tar.gz file available for public access and change the url below.


Now you will have to stack your images. For that, you must be aware that
GSAOI images are highly distorted and that this distortion must be corrected
before stacking. At this moment, the standard tool for distortion correction
and image stacking is called ``disco-stu`` and the most recent version is the
v1.3.4. This package can be found in the link bellow (only available within
Gemini Internal Network for now and requires login):

*  `disco-stu v1.3.4 <https://gitlab.gemini.edu/DRSoftware/disco_stu/repository/v1.3.4/archive.tar.gz>`_

.. Warning::

  The functionality of ``disco-stu`` is being incorporated withing DRAGONS.
  Because of that, you might find unexpected results. Specially in very
  crowded fields where the sky cannot be properly measured. This section
  will be changed in the future.

This package was created to be accessed via command line. Because of that, we
need a few more steps while running it. First, let's import some libraries:

.. code-block:: python
    :linenos:
    :lineno-start: 64

    from collections import namedtuple

    from disco_stu import disco
    from disco_stu.lookups import general_parameters as disco_pars


Then we need to create a special class using :func:`~collections.namedtuple`.
This object will hold information about matching the objects between files:

.. code-block:: python
    :linenos:
    :lineno-start: 68

    MatchInfo = namedtuple(
        'MatchInfo', [
            'offset_radius',
            'match_radius',
            'min_matches',
            'degree'
            ])

We now create objects of ``MatchInfo`` class:

.. code-block:: python
    :linenos:
    :lineno-start: 76

    object_match_info = MatchInfo(
        disco_pars.OBJCAT_ALIGN_RADIUS[0],
        disco_pars.OBJCAT_ALIGN_RADIUS[1],
        None,
        disco_pars.OBJCAT_POLY_DEGREE
    )

    reference_match_info = MatchInfo(
        disco_pars.REFCAT_ALIGN_RADIUS[0],
        disco_pars.REFCAT_ALIGN_RADIUS[1],
        disco_pars.REFCAT_MIN_MATCHES,
        disco_pars.REFCAT_POLY_DEGREE
    )

Now, we simply call the :func:`~disco_stu.disco.disco` function and pass the
position arguments.

.. code-block:: python
    :linenos:
    :lineno-start: 76

    disco.disco(
        infiles=reduce_target.output_filenames,
        output_identifier="my_stacked_image.fits",
        objmatch_info=object_match_info,
        refmatch_info=reference_match_info,
        pixel_scale=disco_pars.PIXEL_SCALE,
    )

This function has many other parameters that can be used to customize this step
but further details are out of the scope of this tutorial. Please, refer to the
`disco-stu GitLab Internal Page <https://gitlab.gemini.edu/DRSoftware/disco_stu>`_
for the corresponding information.


