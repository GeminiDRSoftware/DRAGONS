.. _caldb:

Calibration Database
********************

The ``recipe_system.cal_service`` module includes a ``CalDB`` class (and
subclasses called ``UserDB``, ``LocalDB``, and ``RemoteDB``) that provide
DRAGONS with ways to store and retrieve calibrations. Any number of
databases can be used in a cascading manner, with the first one attached
directly to the ``PrimitivesBASE`` object as the ``caldb`` attribute, and
each subsequent database attached via the ``add_database()`` method.

Methods
=======

Each ``CalDB`` object has the following methods:

* add_database: append a new database to the end of the current cascade
* get_calibrations: retrieve calibrations of a specified type for a
  list of input AstroData objects
* set_calibrations: manually set a calibration for a list of input AD objects
* unset_calibrations: manually remove a calibration for a list of input AD
  objects
* clear_calibrations: clear all manually-set calibrations
* store_calibration: store a calibration for later retrieval via association
  rules


These methods work not just on the database object upon which they are
called, but also subsequent databases in the cascade. So a database's
``get_calibrations`` method will pass on those AD objects for which it
was *unable* to find a calibration and then assemble the complete set of
result into an ordered list. If a processed calibration is passed, then
``store_calibration`` will write the AD object to disk in the appropriate
``calibrations/`` subdirectory and then pass the filename onto each of
the databases; if a processed science frame is passed, then the unmodified
AD object is passed on.


The configuration file
======================

DRAGONS has a global configuration file whose default location is
``~/.dragons/dragonsrc``. This can be overridden by the ``$DRAGONSRC``
environment variable, or specified via the ``config`` (``-c``) argument
of ``reduce``. When the appropriate subclass of ``PrimitivesBASE`` is
instantiated, this config file is read. At the present time, the only
section is related to calibrations and the format should be something
like::

   [calibs]
   databases = ~/.dragons/
     fits.hi.gemini.edu


Databases are listed one per line, with indentation required as shown.
These are parsed with the ``cal_service.parse_databases()`` function and,
after a ``UserDB`` object is attached to the ``PrimitivesBASE``, they are
appended one-by-one in a cascade. A database is assumed to refer to an
instance of a ``LocalDB`` if the file exists or its name contains a
forward slash (but not two consecutive slashes), and a ``RemoteDB`` otherwise.

The default configuation for each database is retrieval only, so the
``storeCalibration`` primitive will *not* automatically add the calibration
to any database. This can be changed by including ``store`` on the line
after the name of the database, but ``get`` also needs to be included.


CalDB subclasses
================

The UserDB class
----------------

The ``UserDB`` class should *always* be the first database in a cascade.
When instantiating the ``PrimitivesBASE`` object, the user calibrations
in the ``ucals`` parameter are passed to this object, and these are returned
if they match the requested calibration type. The ``UserDB`` class also
handles user-defined calibration via a dictionary with entries of the form::

   {(ad.calibration_key(), caltype): filename}


which can be defined via the ``setCalibration()`` primitive. This dictionary
is cached to disk as a pickle in ``calibrations/calindex.pkl``. The ``UserDB``
class is the only one for which the ``set``, ``unset``, and ``clear`` methods
have any effect, while ``store`` does nothing.

Finally, ``UserDB`` also handles the standard MDFs for the instrument, which
it locates via ``p.inst_lookups``.


The LocalDB class
-----------------

The ``LocalDB`` class is a lightweight layer over the ``LocalManager`` class
for retrieving calibrations via a set of calibration association rules,
and adding files stored on disk. Processed science files are not stored by
this class.

This class has additional methods to provide an easy-to-use API. These
are ``add_cal``, ``add_directory``, ``remove_cal``, and ``list_files``.
There is also an ``init`` method to initialize a database but this can be
done automatically when the ``LocalDB`` object is instantiated (and this
is what happens during normal operations).

These methods are also accessible via the command-line ``caldb`` script.
This script creates a ``LocalDB`` object by reading the config file and
therefore requires that there is one and only one local database listed
in that file. Alternatively, a database file can be specified
via the ``-d`` option.


The RemoteDB class
------------------

The ``RemoteDB`` class is used for retrieving and storing calibrations via
URLs and is currently very Gemini-specific. It could be subclassed but it
seems unlikely that another observatory will have an interface that has
enough in common to make subclassing preferable to creating a completely
new class.

This class stores processed calibrations *and* processed science frames.
Storage is only permitted when it is instantiated if the ``store`` flag is
set in the config file *and* the appropriate string is in the ``uploads``
attribute of the ``PrimitivesBASE`` object. This class has a ``store_science``
parameter as well as the ``store_cal`` parameter it shares with ``LocalDB``.
