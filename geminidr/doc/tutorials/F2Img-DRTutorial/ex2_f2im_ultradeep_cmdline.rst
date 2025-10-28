.. ex2_f2im_ultradeep_cmdline.rst

.. _ultradeep_cmdline:

**************************************************************
Example 2 - Deep observation - Using the "reduce" command line
**************************************************************

This chapter will guide you on reducing **Flamingos-2 imaging data** using
command line tools. In this example we reduce Flamingos-2 imaging observation
of a rather sparse field but with the objective of going deep.   We will use
this observation to show and discuss the ``ultradeep`` near-infrared imaging
recipe. Just open a terminal to get started.

The dataset
===========

If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`ultradeep_dataset`

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




Set up the Calibration Service
==============================

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_cmdline`.





Check files
===========

For this example, all the raw files we need are in the same directory called
``../playdata/example2/``. Let us learn a bit about the data we have.

Ensure that you are in the ``playground`` directory and that the ``conda``
environment that includes DRAGONS has been activated.

Let us call the command tool |typewalk|:

.. code-block:: bash

   $ typewalk -d ../playdata/example2/

    directory:  /Users/klabrie/data/tutorials/f2img_tutorial/playdata/example2
         S20200104S0075.fits ............... (F2) (GEMINI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
         S20200104S0076.fits ............... (F2) (GEMINI) (IMAGE) (RAW) (SIDEREAL) (SOUTH) (UNPREPARED)
         ...
         S20200107S0035.fits ............... (CAL) (DARK) (F2) (GEMINI) (RAW) (SOUTH) (UNPREPARED)
         S20200107S0036.fits ............... (CAL) (DARK) (F2) (GEMINI) (RAW) (SOUTH) (UNPREPARED)
         ...
         S20200108S0010.fits ............... (CAL) (F2) (FLAT) (GCALFLAT) (GCAL_IR_OFF) (GEMINI) (IMAGE) (LAMPOFF) (RAW) (SOUTH) (UNPREPARED)
         S20200108S0011.fits ............... (CAL) (F2) (FLAT) (GCALFLAT) (GCAL_IR_OFF) (GEMINI) (IMAGE) (LAMPOFF) (RAW) (SOUTH) (UNPREPARED)
         ...
         S20200111S0159.fits ............... (CAL) (DARK) (F2) (GEMINI) (RAW) (SOUTH) (UNPREPARED)
         S20200111S0160.fits ............... (CAL) (DARK) (F2) (GEMINI) (RAW) (SOUTH) (UNPREPARED)
         ...
    Done DataSpider.typewalk(..)


This command will open every FITS file within the directory passed after the
``-d`` flag (recursively) and will print an unsorted table with the file
names and the associated tags. For example, calibration files will always
have the ``CAL`` tag. Flat images will always have the ``FLAT`` tag. Dark
files will have the ``DARK`` tag. This means that we can start getting to
know a bit more about our data set just by looking at the tags. The output
above was trimmed for presentation.

Note that the K-band flats are showing as LAMPOFF.  The lamp is always on, but
for LAMPOFF flats, the shutter is closed.  In K-band, the heat from the lamp
is sufficient to provide a good exposure with the shutter closed.  Darks are
going to be used for the required no-flux frames.


Create file lists
=================

This data set contains science and calibration frames. For some programs, it
could have different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you. You
have to do it. DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps
with that.  It uses Astrodata tags and "|descriptors|" to select the files and
send the filenames to a text file that can then be fed to "|reduce|".  (See the
|astrodatauser| for information about Astrodata.)

First, navigate to the ``playground`` directory in the unpacked data package::

    cd <path>/f2im_tutorial/playground

Two lists for the darks
-----------------------

Our data set contains two sets of DARK files: some 5-second darks
matching the science data and some 2-second darks matching the flats. If you
did not know the exposure times of the darks, you could send the |dataselect|
results to the |showd| command line tool as follows to get the information:

.. code-block:: bash

   $ dataselect --tags DARK ../playdata/example2/*.fits | showd -d exposure_time
   --------------------------------------------------------
   filename                                   exposure_time
   --------------------------------------------------------
   ../playdata/example2/S20200107S0035.fits             2.0
   ../playdata/example2/S20200107S0036.fits             2.0
   ../playdata/example2/S20200107S0037.fits             2.0
   ../playdata/example2/S20200107S0038.fits             2.0
   ../playdata/example2/S20200107S0039.fits             2.0
   ../playdata/example2/S20200107S0040.fits             2.0
   ../playdata/example2/S20200107S0041.fits             2.0
   ../playdata/example2/S20200107S0049.fits             5.0
   ../playdata/example2/S20200107S0050.fits             5.0
   ../playdata/example2/S20200107S0051.fits             5.0
   ../playdata/example2/S20200107S0052.fits             5.0
   ../playdata/example2/S20200107S0053.fits             5.0
   ../playdata/example2/S20200107S0054.fits             5.0
   ../playdata/example2/S20200107S0055.fits             5.0
   ../playdata/example2/S20200111S0159.fits             5.0
   ../playdata/example2/S20200111S0160.fits             5.0
   ../playdata/example2/S20200111S0161.fits             5.0
   ../playdata/example2/S20200111S0257.fits             2.0
   ../playdata/example2/S20200111S0258.fits             2.0
   ../playdata/example2/S20200111S0260.fits             2.0

The ``|`` is the Unix "pipe" operator and it is used to pass output from
|dataselect| to |showd|.

Let us go ahead and create our two list of darks. The following line creates
a list of dark files that have exposure time of 5 seconds:

.. code-block:: bash

   $ dataselect --tags DARK --expr "exposure_time==5" ../playdata/example2/*.fits -o darks_5s.list

``--expr`` is used to filter the files based on their |descriptors|. Here we are
selecting files with exposure time of 5 seconds. You can repeat the same
command with the other exposure time to get the list of short darks.

.. code-block:: bash

   $ dataselect --tags DARK --expr "exposure_time==2" ../playdata/example2/*.fits -o darks_2s.list


A list for the flats
--------------------
Now let us create the list containing the flat files:

.. code-block:: bash

    $ dataselect --tags FLAT ../playdata/example2/*.fits -o flats.list

We know that our dataset has only one filter (K-red). If our dataset
contained data with more filters, we would have had to use the ``--expr``
option to select the appropriate filter as follows:

.. code-block:: bash

    $ dataselect --tags FLAT --expr "filter_name=='K-red'" ../playdata/example2/*.fits -o flats_Kred.list

.. note::
    Flamingos-2 K-band flat fields are created from lamps-off flats and darks.


A list for the science observations
-----------------------------------

Finally, we want to create a list of the science targets. We are looking for
files that are not calibration frames. To exclude them from our
selection we can use the ``--xtags``, e.g., ``--xtags CAL``.

.. code-block:: bash

    $ dataselect --xtags CAL ../playdata/example2/*.fits -o sci_images.list

Remember that you can use the ``--expr`` option to select targets with different
names (``object``) or exposure times (``exposure_time``), or use it with any
of the datasets |descriptors|.


Create a Master Dark
====================

We start the data reduction by creating a master dark for the science data.
Here is how you reduce the 5 s dark data into a master dark:

.. code-block:: bash

    $ reduce @darks_5s.list

The ``@`` character before the name of the input file is the "at-file" syntax.
More details can be found in the |atfile| documentation.

Because the database was given the "store" option in the ``dragonsrc`` file,
the processed dark will be automatically added to the database at the end of
the recipe.

.. note:: The file name of the output processed dark is the file name of the
    first file in the list with ``_dark`` appended as a suffix.  This the
    general naming scheme used by "|reduce|".

.. note:: If you wish to inspect the processed calibrations before adding them
    to the calibration database, remove the "store" option attached to the
    database in the ``dragonsrc`` configuration file.  You will then have to
    add the calibrations manually following your inspection, eg.

    ``caldb add S20200107S0049_dark.fits``

.. note::
    The master dark will be saved in the same folder where |reduce| was
    called *and* inside the ``./calibrations/processed_dark`` folder. The latter
    location is to cache a copy of the file. This applies to all the processed
    calibration.


Create a Master Flat Field
==========================

The F2 K-red master flat is created from a series of lamp-off exposures and
darks. They should all have the same exposure time. Each flavor is
stacked (averaged), then the dark stack is subtracted from the lamp-off
stack and the result normalized.

We create the master flat field and add it to the calibration manager as
follow:

.. code-block:: bash

    $ reduce @flats_Kred.list @darks_2s.list

It is important to put the flats first in that call.  The recipe is selected
based on the astrodata tags of the first file in the list of inputs.


Reduce the Science Images
=========================
Now that we have the master dark and the master flat, we can tell |reduce|
to process our science data. |reduce| will look at the local database
for calibration files.

We will be running the ``ultradeep`` recipe, the 3-part version.  If you
prefer to run the whole thing in one shot, just call the full recipe with
``-r ultradeep``.

The first part of the ultradeep recipe does the pre-processing, up to and
including the flatfield correction.  This part is identical to what is being
done the in default F2 recipe.

.. code-block:: bash

    $ reduce @sci_images.list -r ultradeep_part1

The outputs are the ``_flatCorrected`` files.

The ``ultradeep_part2`` recipe takes ``_flatCorrected`` images from part 1 as
input and continues the reduction to produce a stacked image. It then
identifies sources in the stack and transfers the object mask (OBJMASK) back
to the individual input images, saving those to disk, ready for part 3.

.. code-block:: bash

    $ reduce S20200104*_flatCorrected.fits -r ultradeep_part2

The outputs are the ``_objmaskTransferred`` files.


Finally, the ``ultradeep_part3`` recipe takes flat-corrected images with
the object masks (``_objmaskTransferred``) as inputs and produces a final stack.

.. code-block:: bash

    $ reduce S20200104*_objmaskTransferred.fits -r ultradeep_part3

The final product file has a ``_image.fits`` suffix.

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


