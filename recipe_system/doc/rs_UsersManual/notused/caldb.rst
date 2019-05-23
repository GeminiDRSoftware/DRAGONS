
caldb
-----

This tools allows you to interact with a local or a remote database where
DRAGONS ``reduce`` will look for calibration files. Its basic usage can be
always checked using the ``--help`` flag.::

    $ caldb --help
    usage: caldb [-h] {config,init,list,add,remove} ...

    Calibration Database Management Tool

    positional arguments:
      {config,init,list,add,remove}
                            Sub-command help
        config              Display configuration info
        init                Create and initialize a new database.
        list                List calib files in the current database.
        add                 Add files to the calibration database. One or more
                            files or directories may be specified.
        remove              Remove files from the calibration database. One or
                            more files may be specified.

    optional arguments:
      -h, --help            show this help message and exit


One, and only one, positional argument should be provided when calling
``caldb``.

If this is the first time that you are using DRAGONS, you have to setup either
a local or a remote database. For a local database, create a file called
``rsys.cfg`` within the ``~/.geminidr/`` directory. The ``~`` means the user's
home folder. You can check if you already have one using the following command:::

    $ cat ~/.geminidr/rsys.cfg
    [calibs]
    standalone = True
    database_dir = ~/.geminidr

If you get an error, you might have to create this directory and/or the file
itself. Its content should be similar to the ones displayed above. The
``standalone`` option tells ``caldb`` if you are using a local database (True)
or a remote database (False). If you ``standalone = True``, you have to set
the ``database_dir`` to an existing directory where the local database will be
stored (by detault, it is the ``~/.geminidr`` itself).

.. todo:: What about remote database?

Once you set this file, you have to make ``caldb`` read it. You do it using the
following command:::

    $ caldb config

    Using configuration file: ~/.geminidr/rsys.cfg

    The active database directory is:  /path/to/.geminidr
    Thus the database file to be used: /path/to/.geminidr/cal_manager.db

    The 'standalone' flag is active, meaning that local calibrations will be used

If everything is fine, you should see the message above.

Now you have to initialize the database. For that, you use:::

    $ caldb init


If you get an error saying that you cannot initialize an existing database,
you can delete the database local file or wipe it using:::

    $ caldb init --wipe

This will wipe out the current existing database. You will lose any information
stored there!

After initializing it, you are ready to add new calibration files. For the
current version, you have to add one file per command:::

    $ caldb add /path/to/calibrations/my_calibration_file.fits
    /path/to/.geminidr

This file will be stored in the database and you can check if the operation
succeeded using the ``list`` argument:::

    $ caldb list
    /path/to/.geminidr
    my_calibration_file.fits        /path/to/calibrations/

If needed, you can remove this file from the database using the following
command::

    $ caldb remove my_calibration_file.fits
    /path/to/.geminidr

.. warning:: If you want/need to update a file that is already stored within
    the database, you will have to remove it and add it again. ``caldb`` has
    no update tool.
