.. 03_api_reduction.rst

.. include:: DRAGONSlinks.txt

.. |github| image:: /_static/img/GitHub-Mark-32px.png
    :scale: 75%


.. _api_data_reduction:

*******************
Reduction using API
*******************

There may be cases where you might be interested in accessing the DRAGONS'
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. In this case, you will need to access
DRAGONS' tools by importing the appropriate modules and packages.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`about_data_set`.

Here is a copy of the table for quick reference.

+---------------+---------------------+--------------------------------+
| Science       || S20170505S0095-110 || Kshort-band, on target, 60 s  |
+---------------+---------------------+--------------------------------+
| Flats         || S20170505S0030-044 || Lamp on, Kshort, for science  |
|               || S20170505S0060-074 || Lamp off, Kshort, for science |
+---------------+---------------------+--------------------------------+
| Standard star || S20170504S0114-117 || Kshort, standard star, 30 s   |
+---------------+---------------------+--------------------------------+

Setting up
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


.. todo @bquint: the gempy auto-api is not being generated anywhere.

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
    logutils.config(file_name='gsaoi_data_reduction.log')


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
    database_dir = ${path_to_my_data}/gsaoiimg_tutorial/playground


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

Next step is to create lists of files that will be used as input to each of the
data reduction steps. Let us start by creating a :class:`list` of all the
FITS files in the directory ``../playdata/``.

.. code-block:: python
    :linenos:
    :lineno-start: 15

    all_files = glob.glob('../playdata/*.fits')

Before you carry on, you might want to do ``print(all_files)`` to check if they
were properly read.

Now we can use the ``all_files`` :class:`list` as an input to
:func:`~gempy.adlibrary.dataselect.select_data`.  The
``dataselect.select_data()`` function signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')


A list for the flats
--------------------
Now you must create a list of FLAT images for each filter. The expression
specifying the filter name is needed only if you have data from multiple
filters. It is not really needed in this case.


.. code-block:: python
    :linenos:
    :lineno-start: 16

    list_of_flats_Ks = dataselect.select_data(
         all_files,
         ['FLAT'],
         [],
         dataselect.expr_parser('filter_name=="Kshort"')
    )


A list for the standard star
----------------------------
For the standard star selection, we use:

.. code-block:: python
    :linenos:
    :lineno-start: 22

    list_of_std_stars = dataselect.select_data(
        all_files,
        [],
        [],
        dataselect.expr_parser('observation_class=="partnerCal"')
    )


Here, we are passing empty lists to the second and the third argument since
we do not need to use the Tags for selection nor for exclusion.


A list for the science data
---------------------------
Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 28

    list_of_science_images = dataselect.select_data(
        all_files,
        [],
        [],
        dataselect.expr_parser('(observation_class=="science" and exposure_time==60.)')
    )

The exposure time is not really needed in this case since there are only
60-second frames, but it shows how you could have two selection criteria in
the expression.


.. _api_process_flat_files:

Create a Master Flat Field
==========================
As explained on the `calibration webpage for GSAOI
<https://www.gemini.edu/sciops/instruments/gsaoi/calibrations>`_,
*dark subtraction is not necessary* since the dark noise level is very low.
Therefore, we can go ahead and start with the master flat.

A GSAOI K-short master flat is created from a series of lamp-on and lamp-off
exposures. Each flavor is stacked, then the lamp-off stack is subtracted from
the lamp-on stack and the result normalized.

We create the master flat field and add it to the calibration manager as
follow:


.. code-block:: python
    :linenos:
    :lineno-start: 34

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats_Ks)
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

Once :meth:`runr()` is finished, we add the master flat to the calibration
manager (line 38).


Reduce Standard Star
====================
The standard star is reduced essentially the same way as the science
target (next section). The processed flat field that we added above to
the local calibration database will be fetched automatically.

.. code-block:: python
    :linenos:
    :lineno-start: 39

    reduce_std = Reduce()
    reduce_std.files.extend(list_of_std_stars)
    reduce_std.runr()

For stacking the sky-subtracted standard star images, the easiest way is
probably to use ``disco_stu``'s command line interface as follow:

::

    $ disco `dataselect *_skySubtracted.fits --expr='observation_class=="partnerCal"'`

If you really want or need to run ``disco_stu``'s API, see the example later
in this chapter where we do just that for the science frames.


.. _api_process_science_files:

Reduce the Science Images
=========================
The science observation uses a dither-on-target with offset-to-sky pattern.
The sky frames from the offset-to-sky position will be automatically detected
and used for the sky subtraction.

The master flat will be retrieved automatically from the local calibration
database.

We use similar commands as before to initiate a new reduction to reduce the
science data:

.. code-block:: python
    :linenos:
    :lineno-start: 42

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_images)
    reduce_target.uparms.append(('skyCorrect:offset_sky', False))
    reduce_target.runr()


.. _api_stack_science_images:

Stack Sky-subtracted Science Images
===================================
The final step is to stack the images. For that, you must be aware that
GSAOI images are highly distorted and that this distortion must be corrected
before stacking. The tool for distortion correction and image stacking is
``disco_stu``.

.. note:: ``disco_stu`` is installed with conda when the standard Gemini
          software installation instructions are followed. To install after the
          fact::

            conda install disco_stu

This package was created to be accessed via command line. Because of that,
the API is not the most polished, and using it requires a fair number of steps.
**If you can use the command line interface, it is recommended that you do so.**
If not, then let's get to work.

First, let's import some libraries:

.. code-block:: python
    :linenos:
    :lineno-start: 45

    from collections import namedtuple

    from disco_stu import disco
    from disco_stu.lookups import general_parameters as disco_pars


Then we need to create a special class using :func:`~collections.namedtuple`.
This object will hold information about matching the objects between files:

.. code-block:: python
    :linenos:
    :lineno-start: 49

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
    :lineno-start: 56

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

Finally, we call the :func:`~disco_stu.disco.disco` function and pass the
arguments.

.. code-block:: python
    :linenos:
    :lineno-start: 69

    disco.disco(
        infiles=reduce_target.output_filenames,
        output_identifier="my_Kshort_stack",
        objmatch_info=object_match_info,
        refmatch_info=reference_match_info,
        pixel_scale=disco_pars.PIXEL_SCALE,
        skysub=False,
    )

This function has many other parameters that can be used to customize this step
but further details are out of the scope of this tutorial.



