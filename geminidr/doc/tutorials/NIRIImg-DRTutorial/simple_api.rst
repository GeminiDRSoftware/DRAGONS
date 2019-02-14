.. simple_api.rst

.. _simple_api:

******************************************
Extended source - Using the "Reduce" class
******************************************

A reduction can be initiated from the command line but it can also be done
programmatically.  The classes and modules of the RecipeSystem can be
accessed directly for those who want to write Python programs to drive their
reduction.  In this section we replicate the previous reduction but using
Python instead of the command line.

The dataset
===========
This is a NIRI imaging observation of the an extended source, a galaxy showing
as a dense field of stars.  The observation sequence uses offset to sky to
monitor it.

The calibrations we use here include:

* Darks for the science and sky offset frames.
* Flats, as a sequence of lamps-on and lamps-off exposures.
* Short darks to use with the flats to create a bad pixel mask.
* A set of standard star observations.

Here are the files that need to be downloaded from the Gemini Observatory
Archive.

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

A note about finding the short darks.  Those are used solely to create a
fresh bad pixel mask (BPM).  In the archive, the calibration association
will not find those darks for you, you will need to search for them
explicitely. To do so,

* Set a date range around the dates of your science observations.
* Set **Instrument** to NIRI.
* Set the **Obs.Type** to DARK.
* Set the exposure time to 1 second.

All the data needed to run this tutorial are found in the tutorial's data
package (KL??? name of the package, with URL).  Download it and unpack it
somewhere convenient.

.. highlight:: bash

::

    cd <somewhere convenient>
    tar xvzf KL???

The datasets are found in the subdirectory ``niriimg_tutorial/playdata``, and we
will work in the subdirectory named ``niriimg_tutorial/playground``.


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

