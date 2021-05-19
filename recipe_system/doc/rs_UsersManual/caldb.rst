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

Configuring local databases for DRAGONS
=======================================
DRAGONS can be configured to use one or more local calibration databases,
which it will query for processed calibrations if these are not specified
directly on the command line or via the API. This is achieve by means of a
configuration file. The default location for this is a file called
``dragonrc`` in a special directory named ``~/.dragons/``, but it can be
changed by use of the ``$DRAGONSRC`` environment variable.  The ``~``
means the user's home directory.  The very first step, to be done only once,
is to create the directory and the configuration file.

::

    $ mkdir ~/.dragons
    $ touch ~/.dragons/dragonsrc

The ``dragonsrc`` file must contain the following lines::

    [calibs]
    databases = ~/.geminidr/cals.db get store  # set this file to whatever you want
      ~/another_directory/another_file.db

Additional databases can be listed, one per line, provided they are indented
as shown above. It is possible to specify only a directory, instead of a
filename, in which case the database file will be called ``cal_manager.db``.
By default each database is configured for retrieval only, and this is what
will happen if only the filename (or directory) is listed. If you wish to
have processed calibrations stored automatically when they are produced by
DRAGONS, you should add the word "store" to that line, as shown. Adding any
flags unsets the "get" flag, so this must be explicitly specified if you
wish to both retrieve *and* store processed calibrations in a database.

When retrieving calibrations, the databases will be searched in order for
each file requiring a calibration, until a suitable calibration is found.
This means the "best" (e.g., closest in time) calibration may not always be
assigned, if an acceptable calibration is found in a database higher in the
list. When storing, each processed calibration will be stored in *every*
database configured to allow storage (i.e., with the "store" flag on its
line in the configuration file).


Using ``caldb`` on the Command Line
===================================
The ``caldb`` tool is used to interact with the local calibration database.
This is where the Recipe System will look for processed calibrations.  For
a reminder of its basic usage, one can always use the ``--help`` flag::

    $ caldb --help
    usage: caldb [-h] {init,list,add,remove} ...

    Calibration Database Management Tool

    positional arguments:
      {config,init,list,add,remove}
                            Sub-command help
        init                Create and initialize a new database.
        list                List calib files in the current database.
        add                 Add files to the calibration database. One or more
                            files or directories may be specified.
        remove              Remove files from the calibration database. One or
                            more files may be specified.

    optional arguments:
      -h, --help            show this help message and exit

Help for each of the sub-commands can be obtained by using the ``--help`` or
``-h`` flag after the appropriate sub-command::

    $ caldb add -h
    usage: caldb add [-h] [-k] [-c CONFIG] [-d DB_PATH] [-v] path [path ...]

    positional arguments:
      path                  FITS file or directory

    optional arguments:
      -h, --help            show this help message and exit
      -k, --walk            If this option is active, directories will be explored
                            recursively. Otherwise, only the first level will be
                            searched for FITS files.
      -c CONFIG, --config CONFIG
                            Path to the config file, if not the default location,
                            or defined by the environment variable.
      -d DB_PATH, --database DB_PATH
                            Path to the database file. Optional if the config file
                            defines a single database.
      -v, --verbose         Give information about the operations being performed.

By default, the database will be determined from the configuration file
(either in its default location, or that indicated by ``$DRAGONSRC``, or
specified with the ``-c`` flag on the command line). If more than one local
database is listed in this file, ``caldb`` will raise an error and exit.
A specific database file can be indicated with the ``-d`` flag, which will
circumvent the configuration file (and, in fact, does not even require there
to be a configuration file).  Since ``caldb`` operates on a *single*
database, the flags in the configuration file are ignored, and the
``caldb add`` command will add the calibration even if the ``store`` flag
is not set.

In the following examples, it is assumed that your configuration file only
lists a single local database, but that need not be true if the ``-d`` flag
is used. Note, however, that this must appear *after* the ``caldb``
sub-command but *before* the filenames of any calibrations to be added to
the database.

To initialize a new database::

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
The above commands are also available in an API, using the ``LocalDB``
class, which takes the filename of the database. This circumvents the
configuration file, which exists to inform DRAGONS of the database
locations and hierarchy. As with the command-line interface, this means
that calibrations will be added even if this database is listed in the
configuration file but the ``store`` flag is not set.

    >>> from recipe_system import cal_service
    >>>
    >>> caldb = cal_service.LocalDB(database_filename)

To add a processed calibration to the database::

    >>> caldb.add_cal('/path/to/master_bias.fits')

If the path is not given, the current directory is assumed.  The addition
of a file to the database is simply the addition of the filename and
its location on the disk.  The file itself *is not stored*.  If the
calibration file is deleted or moved, the database will not know and still
think that the file is there.

It is also possible to add all the files in a given directory to the
database::

    >>> caldb.add_directory('/path/to/calibrations/', walk=False)

where setting the ``walk`` parameter will cause all files in subdirectories
to be added as well.

To see what is in the database::

    >>> for f in caldb.list_files():
    ...     print(f)
    ...
    FileData(name=u'master_bias.fits', path=u'/path/to')

To remove a file from the database::

    >>> caldb.remove_calibration('master_bias.fits')


.. warning:: If a file that is already stored within the database needs
   updating, it will need to be removed and added  again. ``caldb`` has
   no update tool.

To see it used in a complete example along with the other tools see
:ref:`api_example`.
