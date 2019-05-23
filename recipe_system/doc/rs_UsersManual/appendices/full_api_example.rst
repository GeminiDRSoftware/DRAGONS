.. full_api_example.rst

.. _api_example:

****************
Full API Example
****************
Here we put together several of the tools to show how it can all work, from
beginning to end.

1. Import everything we will need.

   ::

    >>> import glob

    >>> from recipe_system.reduction.coreReduce import Reduce
    >>> from recipe_system import cal_service

    >>> from gempy.utils import logutils
    >>> from gempy.adlibrary import dataselect

2. Create the file lists.  One for the darks, one for the flats, one for the
   science target.

   ::

    >>> all_files = glob.glob('../raw/*.fits')

    # 20 second darks.
    >>> expression = 'exposure_time==20'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> darks20s = dataselect.select_data(all_files, ['DARK'], [], parsed_expr)

    # all the flats
    >>> flats = dataselect.select_data(all_files, ['FLAT'])

    # the science data
    >>> expression = 'object=="SN2014J"'
    >>> parsed_expr = dataselect.expr_parser(expression)
    >>> target = dataselect.select_data(all_files, expression=parsed_expr)

3. Set up the calibration manager and database.  First, create or edit the
   ``~/.geminidr/rsys.cfg`` to look like this:

   ::

    [calibs]
    standalone = True
    database_dir = <where_you_want_the_database_to_live>/

   Then configure and initialize the database, and activate the service:

   ::

    >>> caldb = cal_service.CalibrationService()
    >>> caldb.config()
    >>> caldb.init()
    >>> cal_service.set_calservice()

4. Set up the logger.

   ::

    >>> logutils.config(file_name='example.log')

5. Reduce the darks and add the master dark to the calibration database.

   ::

    >>> reduce_darks = Reduce()
    >>> reduce_darks.files.extend(darks20s)
    >>> reduce_darks.runr()

    >>> caldb.add_cal(reduce_darks.output_filenames[0])

6. Reduce the flats and add the master flats to the calibration database.

   ::

    >>> reduce_flats = Reduce()
    >>> reduce_flats.files.extend(flats)
    >>> reduce_flats.runr()

    >>> caldb.add_cal(reduce_flats.output_filenames[0])

7. Reduce the science target, with some input parameter override.

   ::

    >>> reduce_target = Reduce()
    >>> reduce_target.files.extend(target)
    >>> reduce_target.uparms.append(('skyCorrect:scale', False))
    >>> reduce_target.runr()

