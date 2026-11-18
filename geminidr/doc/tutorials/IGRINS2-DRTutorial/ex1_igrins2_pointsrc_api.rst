.. ex1_igrins2_pointsrc_api

.. _igrins2_pointsrc_api:

******************************
Example 1 - Point Source - API
******************************

In this example, we will reduce an IGRINS-2 observation of a CEMP-no star
in a programmatic way with the ``Reduce`` API.

The dither pattern is a standard ABBA sequence.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`igrins2_pointsrc_dataset`

Here is a copy of the table for quick reference.

+--------------------------------+-----------------------------------+
| Science                        || N20260130S0179-182               |
+--------------------------------+-----------------------------------+
| Flats                          || N20260130S0369,373,377,382,385,  |
|                                || 388,391,396,401,407-427          |
+--------------------------------+-----------------------------------+
| Sky for wavelength calibration || N20260130S0021                   |
+--------------------------------+-----------------------------------+
| Telluric                       || N20260130S0183-186               |
+--------------------------------+-----------------------------------+

Setting up
==========
First navigate to your work directory in the unpacked data package.

::

    cd <path>/igrins2_tutorial/playground

The first steps are to import libraries, set up the calibration manager,
and set the logger.

Configuring the interactive interface
-------------------------------------
In ``~/.dragons/``, add the following to the configuration file ``dragonsrc``::

    [interactive]
    browser = your_preferred_browser

The ``[interactive]`` section defines your preferred browser.  DRAGONS will open
the interactive tools using that browser.  The allowed strings are "**safari**",
"**chrome**", and "**firefox**".

Importing libraries
-------------------

.. code-block:: python
    :linenos:

    import glob

    import astrodata
    import gemini_instruments
    from recipe_system.reduction.coreReduce import Reduce
    from gempy.adlibrary import dataselect

The ``dataselect`` module will be used to create file lists for the
biases, the flats, the arcs, the telluric star, and the science observations.
The ``Reduce`` class is used to set up and run the data
reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger.  (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 7

    from gempy.utils import logutils
    logutils.config(file_name='igrins2_tutorial.log')


Set up the Local Calibration Manager
------------------------------------

.. important::  Remember to set up the calibration service.

    Instructions to configure and use the calibration service are found in
    :ref:`cal_service`, specifically the these sections:
    :ref:`cal_service_config` and :ref:`cal_service_api`.

We recommend that you clean up your working directory (``playground``) and
create a fresh calibration database (``caldb.init(wipe=True)``) when you
start a new example.


Create file lists
=================

This data set contains science and calibration frames. For some programs, it
could contain different observed targets and different exposure times depending
on how you like to organize your raw data.

The DRAGONS data reduction pipeline does not organize the data for you.  You
have to do it.  However, DRAGONS provides tools to help you with that.

The first step is to create input file lists.  The tool "|dataselect|" helps.
It uses Astrodata tags and |descriptors| to select the files and
send the filenames to a text file that can then be fed to ``Reduce``.  (See the
|astrodatauser| for information about Astrodata and for a list
of |descriptors|.)


Unbundle the Bundles
--------------------
The IGRINS-2 data raw data files are organized in bundles that contain the
data from the H-band arm and the K-band arm.  The first step is to debundle
the data so that the H-band and K-band files are separated.

.. code-block:: python
    :linenos:
    :lineno-start: 9

    all_raw_files = glob.glob('../playdata/example1/*.fits')
    all_raw_files.sort()
    all_bundles = dataselect.select_data(all_raw_files, ['BUNDLE'])

    reduce_debundle = Reduce()
    reduce_debundle.files.extend(all_bundles)
    reduce_debundle.runr()


A list for all H and K channel files
------------------------------------
This list will be used to create the other lists for the flats, sky for
wavelength calibration, telluric star, and science observations.

.. code-block:: python
    :linenos:
    :lineno-start: 15

    all_files = glob.glob('./*.fits')
    all_files.sort()
    all_hk_files = dataselect.select_data(all_files, ['UNPREPARED'])


Two lists for the flats
-----------------------
The IGRINS-2 flats consist of a series of lamp-on and lamp-off exposures.
For this observation, there are 20 lamp-on and 10 lamp-off exposures.  The
software knows how to combine the lamp-on and lamp-off exposures to create
the master flat.  For each arm, we will put all the flats associated with
the science sequence in one list.

.. code-block:: python
    :linenos:
    :lineno-start: 19

    flatsH = dataselect.select_data(
        all_files,
        tags=['FLAT'],
        xtags=[],
        expression=dataselect.expr_parser('band=="H"')
    )
    flatsK = dataselect.select_data(
        all_files,
        tags=['FLAT'],
        xtags=[],
        expression=dataselect.expr_parser('band=="K"')
    )


Two lists for the zenith sky frame
----------------------------------
The zenith sky frame is used to create the wavelength solution using the
sky emission lines instead of the arc lamp lines.  The zenith sky frame is
generally a single exposure.

.. code-block:: python
    :linenos:
    :lineno-start: 30

    skywlenH = dataselect.select_data(
        all_files,
        tags=['SKY'],
        xtags=[],
        expression=dataselect.expr_parser('band=="H"')
    )
    skywlenK = dataselect.select_data(
        all_files,
        tags=['SKY'],
        xtags=[],
        expression=dataselect.expr_parser('band=="K"')
    )


Two lists for the telluric
--------------------------
The IGRINS-2 telluric standards are tagged STANDARD and can be easily separated
from the science observations.

.. code-block:: python
    :linenos:
    :lineno-start: 41

    telluricH = dataselect.select_data(
        all_files,
        tags=['STANDARD'],
        xtags=[],
        expression=dataselect.expr_parser('band=="H"')
    )
    telluricK = dataselect.select_data(
        all_files,
        tags=['STANDARD'],
        xtags=[],
        expression=dataselect.expr_parser('band=="K"')
    )


Two lists for the science observations
--------------------------------------
The science observations are everything not a calibration, ie. not tagged
``CAL``.

.. code-block:: python
    :linenos:
    :lineno-start: 52

    scienceH = dataselect.select_data(
        all_files,
        tags=[],
        xtags=['CAL'],
        expression=dataselect.expr_parser('band=="H"')
    )
    scienceK = dataselect.select_data(
        all_files,
        tags=[],
        xtags=['CAL'],
        expression=dataselect.expr_parser('band=="K"')
    )


Master Flat Field
=================
The IGRINS-2 flats are obtained at the end of the night.  They are a series
of lamp-on and lamp-off exposures.  The software will combine the
lamp-on and lamp-off exposures and create the master flat.

