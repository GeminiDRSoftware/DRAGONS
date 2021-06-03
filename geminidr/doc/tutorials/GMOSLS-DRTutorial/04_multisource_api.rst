.. multisource_api.rst

.. include:: DRAGONSlinks.txt

.. _multisource_api:

*************************************************************
Example 1-B: Multi-source Longslit - Using the "Reduce" class
*************************************************************

A reduction can be initiated from the command line as shown in
:ref:`multisource_cmdline` and it can also be done programmatically as we will
show here.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this example we replicate the command line reduction from
Example 1-A, this time using the Python interface instead of the command line.
Of course what is shown here could be packaged in modules for greater
automation.

The dataset
===========
If you have not already, download and unpack the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`datamultisource`.

Here is a copy of the table for quick reference.

+---------------------+---------------------------------+
| Science             || N20180526S1024-1025 (650 nm)   |
|                     || N20180526S1028-1029 (660 nm)   |
+---------------------+---------------------------------+
| Science biases      || N20180525S0292-296             |
|                     || N20180527S0848-852             |
+---------------------+---------------------------------+
| Science flats       || N20180526S1023 (650 nm)        |
|                     || N20180526S1026 (650 nm)        |
|                     || N20180526S1027 (660 nm)        |
|                     || N20180526S1030 (660 nm)        |
+---------------------+---------------------------------+
| Science arcs        || N20180527S0001 (650 nm)        |
|                     || N20180527S0002 (660 nm)        |
+---------------------+---------------------------------+
| Standard (Feige 34) || N20180423S0024 (650 nm)        |
+---------------------+---------------------------------+
| Standard biases     || N20180423S0148-152             |
|                     || N20180422S0144-148             |
+---------------------+---------------------------------+
| Standard flats      || N20180423S0025 (650nm)         |
+---------------------+---------------------------------+
| Standard arc        || N20180423S0110 (650 nm)        |
+---------------------+---------------------------------+


Setting up
==========
First, navigate to your work directory in the unpacked data package.

The first steps are to import libraries, set up the calibration manager,
and set the logger.


Importing libraries
-------------------

.. code-block:: python
    :linenos:

    import glob

    import astrodata
    import gemini_instruments
    from recipe_system.reduction.coreReduce import Reduce
    from recipe_system import cal_service
    from gempy.adlibrary import dataselect

The ``dataselect`` module will be used to create file lists for the
darks, the flats and the science observations. The ``cal_service`` package
is our interface to the local calibration database. Finally, the
``Reduce`` class is used to set up and run the data reduction.


Setting up the logger
---------------------
We recommend using the DRAGONS logger.  (See also :ref:`double_messaging`.)

.. code-block:: python
    :linenos:
    :lineno-start: 8

    from gempy.utils import logutils
    logutils.config(file_name='niri_tutorial.log')


Set up the Local Calibration Manager
------------------------------------
DRAGONS comes with a local calibration manager and a local, light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows the ``Reduce`` instance to make requests for matching
**processed** calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/``, edit the configuration file ``rsys.cfg`` as follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/gmosls_tutorial/playground

This tells the system where to put the calibration database, the
database that will keep track of the processed calibration we are going to
send to it.

.. note:: The tilde (``~``) in the path above refers to your home directory.
    Also, mind the dot in ``.geminidr``.

The calibration database is initialized and the calibration service is
configured like this:

.. code-block:: python
    :linenos:
    :lineno-start: 10

    caldb = cal_service.CalibrationService()
    caldb.config()
    caldb.init()

    cal_service.set_calservice()

The calibration service is now ready to use.  If you need more details,
check the "|caldb|" documentation in the Recipe System User Manual.


Create file lists
=================
The next step is to create input file lists.  The module ``dataselect`` helps
with that.  It uses Astrodata tags and |descriptors| to select the files and
store the filenames to a Python list that can then be fed to the ``Reduce``
class. (See the |astrodatauser| for information about Astrodata and for a list
of |descriptors|.)

The first list we create is a list of all the files in the ``playdata``
directory.

.. code-block:: python
    :linenos:
    :lineno-start: 16

    all_files = glob.glob('../playdata/*.fits')
    all_files.sort()

We will search that list for files with specific characteristics.  We use
the ``all_files`` :class:`list` as an input to the function
``dataselect.select_data()`` .  The function's signature is::

    select_data(inputs, tags=[], xtags=[], expression='True')

We show several usage examples below.


Two lists for the biases
------------------------
