.. caldb.rst

.. _caldb:

**************************
Local Calibration Database
**************************

The Recipe System has a system to retrieve processed calibration
automatically.  This system must work with a Calibration Manager.
Currently, only one public Calibration Manager is available, the Gemini
Calibration Manager, ``GeminiCalMgr``.  This must be installed as a
DRAGONS dependency; a conda install will take care of that (see
:ref:`install`).

The Calibration Manager contains the calibration association rules and
database access hooks.  The Gemini Calibration Manager uses exactly the
same calibration association rules as the Gemini Observatory Archive (GOA).

The calibration facility requires a database.  The Recipe System's
``caldb`` application helps the user configure and create a local, lightweight
``sqlite`` database, and add or remove calibration files to and from that
database.

In this chapter, we explain how to use ``caldb`` to add processed
calibrations that the Recipe System will pick up when needed.

.. note:: We intend to improve the Calibration Manager side of things
          to make expanding the association rules for new instruments or
          non-Gemini instruments feasible.

.. _config_caldb:

Configuring ``caldb``
=====================
The first time ``caldb`` is used for a project, either via command line or
API, it needs to be configured and initialized.  The configuration is
stored in a text file in a special directory named ``~/.geminidr/``, in a
file called ``rsys.cfg``.  The ``~`` means the user's home directory.  The
very first step, to be done only once, is to create the directory and the
configuration file.

::

    $ mkdir ~/.geminidr
    $ touch ~/.geminidr/rsys.cfg

The ``rsys.cfg`` file must contain the following lines::

    [calibs]
    standalone = True
    database_dir = ~/.geminidr   # set this path to whatever you want.

The ``standalone`` option tells ``caldb`` if you are using a local database
when it is set to True.  ``standalone = False`` is used only internally at
Gemini when using the internal data manager.

The ``database_dir`` parameter points to the directory hosting the calibration
database.  The database name is always ``cal_manager.db``, this cannot be set,
only the directory where it lives.  It is possible to have more than one
database as long as they are in different directory.  Which one will be picked
up will be set through the ``database_dir`` parameter in ``rsys.cfg``.


Using ``caldb`` on the Command Line
===================================
The ``caldb`` tool is used to interact with the local calibration database.
This is where the Recipe System will look for processed calibrations.  For
a reminder of its basic usage, one can always use the ``--help`` flag::

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

There can be only one positional argument given to ``caldb``, this means only
one file at a time can be added or removed from the database.

Once the configuration file is in place (see :ref:`config_caldb`), one can
verify the configuration by doing::

    $ caldb config

    Using configuration file: ~/.geminidr/rsys.cfg
    Active database directory: /Users/username/.geminidr
    Database file: /Users/username/.geminidr/cal_manager.db

    The 'standalone' flag is active, meaning that local calibrations will be used

To initialize a new database with the selected configuration::

    $ caldb init

Once the database is initialized (created), it is ready for use.

To add a file::

    $ caldb add /path/to/master_bias.fits

If the path is not given, the current directory is assumed.  The addition
of a file to the database is simply the addition of the filename and
its location on the disk.  The file itself *is not stored*.  If the
calibration file is deleted or moved, the database will not know and still
think that the file is there.

To see what is in the database::

    $ caldb list
    master_bias.fits    /path/to/

To remove a file from the database::

    $ caldb remove master_bias.fits

.. warning:: If a file that is already stored within the database needs
   updating, it will need to be removed and added  again. ``caldb`` has
   no update tool.

To see ``caldb`` used in a complete example along with the other tools see
:ref:`commandline_example`.


Using the ``caldb`` API
=======================
Before being usable in a Python program, the local calibration manager
must be configured.  This cannot be done from the API.  See
:ref:`config_caldb` for instructions.

The calibration database is initialized and the configuration are read into the the calibration service as follow::

    >>> from recipe_system import cal_service
    >>>
    >>> caldb = cal_service.CalibrationService()
    >>> caldb.config()
    >>> caldb.init()
    >>> cal_service.set_calservice()

The calibration service is then ready to use.  This must be done before
``Reduce`` is instantiated.

To add a processed calibration to the database::

    >>> caldb.add_cal('/path/to/master_bias.fits')

If the path is not given, the current directory is assumed.  The addition
of a file to the database is simply the addition of the filename and
its location on the disk.  The file itself *is not stored*.  If the
calibration file is deleted or moved, the database will not know and still
think that the file is there.

To see what is in the database::

    >>> for f in caldb.list_files():
    ...     print(f)
    ...
    FileData(name=u'master_bias.fits', path=u'/path/to')

To remove a file from the database::

    >>> caldb.remove_cal('master_bias.fits')


.. warning:: If a file that is already stored within the database needs
   updating, it will need to be removed and added  again. ``caldb`` has
   no update tool.

To see it used in a complete example along with the other tools see
:ref:`api_example`.
