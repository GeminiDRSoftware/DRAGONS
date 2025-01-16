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

3. Set up the calibration manager and database, by creating or editing the
   ``~/.dragons/dragonsrc`` file as indicated below.

   ::

    [calibs]
    databases = <path_to>/redux_dir/dragons.db get store

   In principle, if you configure the database to automatically store
   calibrations as they are produced, there is no need for you to interact
   with it, as it will automatically be initialized when the reduction
   starts. [#fn1]_ However, you may want to inspect its contents, or possibly
   delete files from it, in which case you should create a python object to
   allow you to interact directly with it, as follows (the final ``init()``
   call is optional unless you want to add calibrations to the database
   before running ``reduce``).

   ::

   >>> from recipe_system import cal_service
   >>> caldb = cal_service.set_local_database()
   >>> caldb.init()


4. Set up the logger.

   ::

    >>> logutils.config(file_name='example.log')

5. Reduce the darks and add the master dark to the calibration database.

   ::

    >>> reduce_darks = Reduce()
    >>> reduce_darks.files.extend(darks20s)
    >>> reduce_darks.runr()

   The resultant processed dark will automatically be added to the database
   provided the ``store`` flag is set. The log will indicate whether the
   file was added or not. If not, it can be manually added::

    >>> caldb.add_cal(reduce_darks.output_filenames[0])

6. Reduce the flats and add the master flat to the calibration database.

   ::

    >>> reduce_flats = Reduce()
    >>> reduce_flats.files.extend(flats)
    >>> reduce_flats.runr()

   Again, the processed flat need only be added manually if the ``store``
   flag was not set::

    >>> caldb.add_cal(reduce_flats.output_filenames[0])

7. Reduce the science target, with some input parameter override.

   ::

    >>> reduce_target = Reduce()
    >>> reduce_target.files.extend(target)
    >>> reduce_target.uparms['skyCorrect:scale'] = False
    >>> reduce_target.runr()


.. rubric:: Footnotes

.. [#fn1] You should always specify the absolute path to the location of
          the database file, using ``~`` as shorthand for your home directory
          if you wish.
