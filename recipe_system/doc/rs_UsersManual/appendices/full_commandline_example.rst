.. full_commandline_example.rst

.. _commandline_example:

*************************
Full Command Line Example
*************************
Here we put together several of the tools to show how it can all work, from
beginning to end.

1. First create the lists.  One for the darks, one for the flats, and one for
   the science target.

   ::

    $ dataselect ../raw/*.fits --tags DARK --expr='exposure_time==20' -o darks20s.lis
    $ dataselect ../raw/*.fits --tags FLAT -o flats.lis
    $ dataselect ../raw/*.fits --expr='object=="SN2014J"' -o target.lis

2. Set the calibration manager and database.  First, create or edit the
   ``~/.dragons/dragonsrc`` to look like this:

   ::

    [calibs]
    databases = <path_to>/redux_dir/dragons.db get store

   Then initialize the calibration database.

   ::

    $ caldb init

3. Reduce the darks and add the master dark to the calibration database.

   ::

    $ reduce @darks20s.lis

4. Reduce the flats and add the master flat to the calibration database.

   ::

    $ reduce @flats.lis

5. Reduce the science target, with some input parameter override.

   ::

    $ reduce @target.lis -p skyCorrect:scale=False
