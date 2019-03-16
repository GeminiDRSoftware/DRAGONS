.. extended_api.rst

.. _extended_api:

*******************************************************
Example 1-B: Extended source - Using the "Reduce" class
*******************************************************

A reduction can be initiated from the command line as shown in
:ref:`extended_cmdline` and it can also be done programmatically as we will
show here.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this example we replicate the command line reduction from
Example 1-A, this time using the Python interface instead of the command line.
Of course what is shown here could be packaged in modules for full automation.


The dataset
===========
If you have not already, download and unpackage the tutorial's data package.
Refer to :ref:`datasetup` for the links and simple instructions.

The dataset specific to this example is described in:

    :ref:`dataextended`.

Here is a copy of the table for quick reference.

+---------------+--------------------------------------------+
| Science       || N20160102S0270-274 (on-target)            |
|               || N20160102S0275-279 (on-sky)               |
+---------------+--------------------------------------------+
| Science darks || N20160102S0423-432 (20 sec, like Science) |
+---------------+--------------------------------------------+
| Flats         || N20160102S0373-382 (lamps-on)             |
|               || N20160102S0363-372 (lamps-off)            |
+---------------+--------------------------------------------+
| Short darks   || N20160103S0463-472                        |
+---------------+--------------------------------------------+
| Standard star || N20160102S0295-299                        |
+---------------+--------------------------------------------+



Setting up
==========
cd to playground, then start python
astrodata, gemini_instruments
what do I need from recipe_system?
caldb

::

    % cd <path>/playground
    % python

::

    >>> from __future__ import print_function
    >>> import astrodata
    >>> import gemini_instruments
    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> from recipe_system.cal_service import CalibrationService
    >>> from recipe_system.cal_service import set_calservice
    >>> from gempy.utils import logutils
    >>> from gempy.adlibrary import dataselect

    >>> import glob



Create file lists
=================

Two lists for the darks
-----------------------

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> all_darks = dataselect.select_data(all_files, ['DARK'])
    >>> for dark in all_darks:
    ...     ad = astrodata.open(dark)
    ...     print(dark, '  ', ad.exposure_time())

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> expression = 'exposure_time==1'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks1s = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)

    >>> expression = 'exposure_time==20'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks20s = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)


A list for the flats
--------------------

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> flats = dataselect.select_data(all_files, ['FLAT'])


A list for the standard star
----------------------------

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> expression = 'object=="FS 17"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> stdstar = dataselect.select_data(all_files, expression=parsed_expr)


A list for the science observations
-----------------------------------

::

    >>> all_files = glob.glob('../playdata/*.fits')
    >>> expression = 'object!="FS 17"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> has_tags = ['IMAGE']
    >>> has_not_tags = ['FLAT']
    >>> target = dataselect.select_data(all_files, has_tags, has_not_tags,
    ...                                 expression=parsed_expr)


Set up the Local Calibration Manager
====================================
DRAGONS comes with a local calibration manager and local light weight database
that uses the same calibration association rules as the Gemini Observatory
Archive.  This allows the ``Reduce`` instance to make requests for matching
**processed** calibrations when needed to reduce a dataset.

Let's set up the local calibration manager for this session.

In ``~/.geminidr/, edit the configuration file ``rsys.cfg`` as follow::

    [calibs]
    standalone = True
    database_dir = <where_the_data_package_is>/niriimg_tutorial/playground

This simply tells the system where to put the calibration database, the
database that will keep track of the processed calibration we are going to
send to it.

::

    >>> from recipe_system.cal_service import CalibrationService
    >>> caldb = CalibrationService()
    >>> caldb.config()

    >>> caldb.init()

    >>> from recipe_system.cal_service import set_calservice
    >>> set_calservice()



Reduce the data
===============

Set up the logging
------------------

::

    >>> from gempy.utils import logutils
    >>> logutils.config(file_name='niri_tutorial.log')

Master Dark
-----------

::

    >>> reduce_darks = Reduce()
    >>> reduce_darks.files.extend(darks1s)
    >>> reduce_darks.runr()

    >>> caldb.add_cal(reduce_darks.output_filenames[0])

::

    >>> reduce_darks = Reduce()
    >>> reduce_darks.files.extend(darks20s)
    >>> reduce_darks.runr()

    >>> caldb.add_cal(reduce_darks.output_filenames[0])



Bad Pixel Mask
--------------

::

    >>> reduce_bpm = Reduce()
    >>> reduce_bpm.files.extend(flats)
    >>> reduce_bpm.files.extend(darks1s)
    >>> reduce_bpm.recipename = 'makeProcessedBPM'
    >>> reduce_bpm.runr()

    >>> print(reduce_bpm.output_filenames)

    >>> bpm = reduce_bpm.output_filenames[0]

Master Flat Field
-----------------

::

    >>> reduce_flats = Reduce()
    >>> reduce_flats.files.extend(flats)
    >>> reduce_flats.uparms = [('addDQ:user_bpm', bpm)]
    >>> reduce_flats.runr()

Standard Star
-------------

::

    >>> reduce_std = Reduce()
    >>> reduce_std.files.extend(stdstar)
    >>> reduce_std.uparms =     [('addDQ:user_bpm', bpm)]
    >>> reduce_std.uparms.append(('darkCorrect:do_dark', False))
    >>> reduce_std.runr()


Science Observations
--------------------

::

    >>> reduce_target = Reduce()
    >>> reduce_target.files.extend(target)
    >>> reduce_target.uparms = [('addDQ:user_bpm', bpm)]
    >>> reduce_target.uparms.append(('skyCorrect:scale', False))
    >>> reduce_target.runr()

