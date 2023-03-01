.. 03_api_reduction.rst

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

.. note:: A master dark is not needed for GSAOI.  The dark current is very low.


Setting up
==========

Importing Libraries
-------------------

We first import the necessary modules and classes:

.. code-block:: python
    :linenos:

    import glob

    from gempy.adlibrary import dataselect
    from recipe_system import cal_service
    from recipe_system.reduction.coreReduce import Reduce


:mod:`glob` is a Python built-in package. It will be used to return a
:class:`list` with the input file names.


.. todo @bquint: the gempy auto-api is not being generated anywhere.

:mod:`~gempy.adlibrary.dataselect` will be used to create file lists for the
darks, the flats and the science observations. The
:mod:`~recipe_system.cal_service` package is our interface with the
calibration databases. Finally, the
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

First, check that you have already a ``dragonsrc`` file inside the
``~/.dragons/``. It should contain:

.. code-block:: none

    [calibs]
    databases = ${path_to_my_data}/gsaoiimg_tutorial/playground/cal_manager.db get store


This tells the system where to put the calibration database. This
database will keep track of the processed calibrations as we add them
to it. The ``store`` option in the database line above indicates that calibrations
will be automatically added to the database as they are produced, without having to
explicitly add them to the database by running ``caldb add``. 

.. note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.dragons``.

The calibration database is initialized and the calibration service is
configured as follow:

.. code-block:: python
    :linenos:
    :lineno-start: 10

    caldb = cal_service.set_local_database()
    caldb.init()

The calibration service is now ready to use. If you need more details,
check the |caldb| section in the |RSUser|.

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
    all_files.sort()

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
    :lineno-start: 17

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
    :lineno-start: 23

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
    :lineno-start: 29

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
    :lineno-start: 35

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats_Ks)
    reduce_flats.runr()

    caldb.add_cal(reduce_flats.output_filenames[0])

Once :meth:`runr()` is finished, we add the master flat to the calibration
manager (line 39).


Reduce Standard Star
====================
The standard star is reduced essentially the same way as the science
target (next section). The processed flat field that we added above to
the local calibration database will be fetched automatically.

.. code-block:: python
    :linenos:
    :lineno-start: 40

    reduce_std = Reduce()
    reduce_std.files.extend(list_of_std_stars)
    reduce_std.runr()

.. note:: ``Reduce`` will automatically align and stack the images. 
      Therefore, it is no longer necessary to use the ``disco_stu`` tool for GSAOI
      data.



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
    :lineno-start: 43

    reduce_target = Reduce()
    reduce_target.files.extend(list_of_science_images)
    reduce_target.uparms.append(('skyCorrect:offset_sky', False))
    reduce_target.runr()

This will generate flat corrected files, align them,
stack them, and orient them such that North is up and East is left. The final
image will have the name of the first file in the set, with the suffix ``_image``.
The on-target files are the ones that have been flat corrected (``_flatCorrected``),
and scaled (``_countsScaled``).  There should be nine of these.


.. figure:: _static/img/S20170505S0095_image.png
   :align: center

   S20170505S0095 - Final flat corrected, aligned, and stacked image

The figure above shows the final flat-corrected, aligned, and stacked frame.
For absolute distortion correction and astrometry, ``Reduce`` can use a
reference catalog provided by the user.  Without a reference catalog, like
above, only the relative distortion between the frames is accounted for. 

The output stack units are in electrons (header keyword BUNIT=electrons).
The output stack is stored in a multi-extension FITS (MEF) file.  The science
signal is in the "SCI" extension, the variance is in the "VAR" extension, and
the data quality plane (mask) is in the "DQ" extension.

