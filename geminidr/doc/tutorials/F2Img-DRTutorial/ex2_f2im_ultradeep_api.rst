.. ex2_f2im_ultradeep_api.rst

.. _ultradeep_api:

**************************************************************************
Example 2 - Deep observation - Using the "Reduce" class
**************************************************************************

There may be cases where you would be interested in accessing the DRAGONS
Application Program Interface (API) directly instead of using the command
line wrappers to reduce your data. Here we show you how to do the same
reduction we did in the previous chapter, :ref:`ultradeep_cmdline`,
but using the API.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`ultradeep_dataset`.

Here is a copy of the table for quick reference.

+---------------+---------------------+-----------------------+
| Science       || S20200104S0075-092 | K-red, 5 s            |
+---------------+---------------------+-----------------------+
| Darks         || S20200107S0035-041 | 2 s, darks for flats  |
|               || S20200111S0257-260 | 2 s, darks for flats  |
|               +---------------------+-----------------------+
|               || S20200107S0049-161 | 5 s, for science dat  |
+---------------+---------------------+-----------------------+
| Flats         || S20200108S0010-019 | 2 s, Lamp On, K-red   |
+---------------+---------------------+-----------------------+

Setting Up
==========
First, navigate to your work directory in the unpacked data package.

::

    cd <path>/f2im_tutorial/playground

The first steps are to import libraries, set up the calibration manager,
and set the logger.

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
    logutils.config(file_name='f2im_data_reduction.log')



Setting up the Calibration Service
----------------------------------

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_api`.




Create list of files
====================

The next step is to create input file lists. The module ``dataselect`` helps
with that.  It uses Astrodata tags and |descriptors| to select the files and
store the filenames to a Python list that can then be fed to the ``Reduce``
class. (See the |astrodatauser| for information about Astrodata and for a list
of |descriptors|.)

The first list we create is a list of all the files in the ``playdata/example2/``
directory.

.. code-block:: python
    :linenos:
    :lineno-start: 9

    all_files = glob.glob('../playdata/example2/*.fits')
    all_files.sort()

The :meth:`~list.sort` method simply re-organize the list with the file names
and is an optional, but a recommended step. Before you carry on, you might want to do
``print(all_files)`` to check if they were properly read.

We will search that list for files with specific characteristics.  We use
the ``all_files`` :class:`list` as an input to the function
``dataselect.select_data()`` .  The function's signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')

We show several usage examples below.


Two lists for the darks
-----------------------
We select the files that will be used to create a master dark for
the science observations, those with an exposure time of 5 seconds.

.. code-block:: python
    :linenos:
    :lineno-start: 11

    dark_files_5s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==5')
    )

Above we are requesting data with tags ``F2``, ``DARK``, and ``RAW``, though
since we only have F2 raw data in the directory, ``DARK`` would be sufficient
in this case. We are not excluding any tags, as represented by the empty
list ``[]``.

.. note::  All expressions need to be processed with ``dataselect.expr_parser``.

We repeat the same syntax for the 2-second darks:

.. code-block:: python
    :linenos:
    :lineno-start: 17

    dark_files_2s = dataselect.select_data(
        all_files,
        ['F2', 'DARK', 'RAW'],
        [],
        dataselect.expr_parser('exposure_time==2')
    )

A list for the flats
--------------------
Now you must create a list of FLAT images for each filter. The expression
specifying the filter name is needed only if you have data from multiple
filters. It is not really needed in this case.

.. code-block:: python
    :linenos:
    :lineno-start: 23

    list_of_flats_Kred = dataselect.select_data(
         all_files,
         ['FLAT'],
         [],
         dataselect.expr_parser('filter_name=="K-red"')
    )


A list for the science data
---------------------------
Finally, the science data can be selected using:

.. code-block:: python
    :linenos:
    :lineno-start: 29

    list_of_science_images = dataselect.select_data(
        all_files,
        ['F2'],
        [],
        dataselect.expr_parser('(observation_class=="science" and filter_name=="K-red")')
    )

The filter name is not really needed in this case since there are only Y-band
frames, but it shows how you could have two selection criteria in
the expression.


Create a Master Dark
====================

We first create the master dark for the science targe.The master biases
will be automatically added to the local calibration manager when the "store"
parameter is present in the ``.dragonsrc`` configuration file.

The name of the output master dark is
``S20200107S0049_dark.fits``. The output is written to disk and its name is
stored in the Reduce instance. The calibration service expects the name of a
file on disk.

.. code-block:: python
    :linenos:
    :lineno-start: 35

    reduce_darks = Reduce()
    reduce_darks.files.extend(dark_files_5s)
    reduce_darks.runr()

The ``Reduce`` class is our reduction
"controller". This is where we collect all the information necessary for
the reduction. In this case, the only information necessary is the list of
input files which we add to the ``files`` attribute. The ``runr`` method is
where the recipe search is triggered and where it is executed.

.. note:: The file name of the output processed dark is the file name of the
    first file in the list with _dark appended as a suffix. This is the general
    naming scheme used by the ``Recipe System``.

.. note:: If you wish to inspect the processed calibrations before adding them
    to the calibration database, remove the "store" option attached to the
    database in the ``dragonsrc`` configuration file.  You will then have to
    add the calibrations manually following your inspection, eg.

   .. code-block::

        caldb.add_cal(reduce_darks.output_filenames[0])



Create a Master Flat Field
==========================
The F2 K-red master flat is created from a series of lamp-off exposures and
darks. They should all have the same exposure time. Each flavor is
stacked (averaged), then the dark stack is subtracted from the lamp-off
stack and the result normalized.

We create the master flat field and add it to the calibration manager as follows:

.. code-block:: python
    :linenos:
    :lineno-start: 38

    reduce_flats = Reduce()
    reduce_flats.files.extend(list_of_flats_Kred)
    reduce_flats.files.extend(dark_files_2s)
    reduce_flats.runr()

It is important to put the flats first in that call.  The recipe is selected
based on the astrodata tags of the first file in the list of inputs.


Reduce the Science Images
=========================
The science observation uses a dither-on-target pattern. The sky frames will
be derived automatically for each science frame from the dithered frames.

The master dark and the master flat will be retrieved automatically from the
local calibration database.

We will be running the ``ultradeep`` recipe, the 3-part version.  If you
prefer to run the whole thing in one shot, just call the full recipe with
``reduce_target.recipename = 'ultradeep'``.

The first part of the ultradeep recipe does the pre-processing, up to and
including the flatfield correction.  This part is identical to what is being
done the in default F2 recipe.


.. code-block:: python
    :linenos:
    :lineno-start: 42

    reduce_target = Reduce()
    reduce_target.files = list_of_science_images
    reduce_target.recipename = 'ultradeep_part1'
    reduce_target.runr()

The outputs are the ``_flatCorrected`` files.  The list is stored in
``reduce_target.output_filenames`` which we can pass to the next call.

The ``ultradeep_part2`` recipe takes ``_flatCorrected`` images from part 1 as
input and continues the reduction to produce a stacked image. It then
identifies sources in the stack and transfers the object mask (OBJMASK) back
to the individual input images, saving those to disk, ready for part 3.

.. code-block:: python
    :linenos:
    :lineno-start: 46

    reduce_target.files = reduce_target.output_filenames
    reduce_target.recipename = 'ultradeep_part2'
    reduce_target.runr()

The outputs are the ``_objmaskTransferred`` files.

Finally, the ``ultradeep_part3`` recipe takes flat-corrected images with
the object masks (``_objmaskTransferred``) as inputs and produces a final stack.

.. code-block:: python
    :linenos:
    :lineno-start: 49

    reduce_target.files = reduce_target.output_filenames
    reduce_target.recipename = 'ultradeep_part3'
    reduce_target.runr()

The final product file will have a ``_image.fits`` suffix.

The output stack units are in electrons (header keyword BUNIT=electrons).
The output stack is stored in a multi-extension FITS (MEF) file.  The science
signal is in the "SCI" extension, the variance is in the "VAR" extension, and
the data quality plane (mask) is in the "DQ" extension.

For this dataset the benefit of the ultradeep recipe is subtle.  Below we
show a zoomed-in section of the final image **when the complete set of 156 images
is used**.  The image on the left is from the default recipe, the one on the
right is from the ultradeep recipe.

.. image:: _graphics/default.png
   :width: 325
   :alt: default recipe

.. image:: _graphics/ultradeep.png
   :width: 325
   :alt: ultradeep recipe

Looking very carefully, it is possible to see weak blotching in the default
recipe image (left) that does dissappear when the ultradeep recipe is used.
Even using the full set, it is still subtle.  Therefore, we recommend the
use of the ultradeep recipe only when you actually needed or when the blotching
is more severe.  The blotching is expected to be more severe in crowded fields.
