#
#                                                                        DRAGONS
#
#                                                                    cal_service
# ------------------------------------------------------------------------------
from os import path
import shlex
import warnings
from importlib import import_module

from ..config import globalConf, load_config

from .userdb import UserDB
from .localdb import LocalDB
from .remotedb import RemoteDB

# ------------------------------------------------------------------------------
# BEGIN Setting up the calibs section for config files
CONFIG_SECTION = 'calibs'

# END Setting up the calibs section for config files
# ------------------------------------------------------------------------------


def get_calconf():
    try:
        return globalConf[CONFIG_SECTION]
    except KeyError:
        # This will happen if CONFIG_SECTION has not been defined in any
        # config file (shouldn't happen if the user has called load_config()
        pass


def get_db_path_from_config():
    """
    Read the path of the local database specified in the config file. An
    error will be raised if there is no such database, or more than one.
    This function is used by the "caldb" script and the set_local_database()
    function here.

    Parameters
    ----------
    config: str
        name of the configuration file

    Returns
    -------
    db_path : str
        the path to the local database file
    """
    if not globalConf.sections():
        raise OSError("Cannot read config file.")
    databases = parse_databases()
    db_path = None
    for db in databases:
        if db[0] == LocalDB:
            if db_path is None:
                db_path = db[1]
            else:
                raise ValueError("Multiple local database files are listed "
                                 "in the config file.")
    if db_path is None:
        raise ValueError("No local database file is listed in the config file.")
    return db_path


def init_calibration_databases(inst_lookups=None, procmode=None,
                               ucals=None, upload=None):
    """
    Initialize the calibration databases for a PrimitivesBASE object.

    Parameters
    ----------
    inst_lookups : str
        local of the instrument lookups package (for the MDF lookup table)
    ucals : dict
        user calibrations
    upload : list
        things to upload (we're concerned about "calibs" and "science")

    Returns
    -------
    A UserDB object, possibly linked to additional CalDB objects
    """
    # Read the mdf_dict file and create an actual dict with the complete
    # paths to each of the MDF files
    try:
        masks = import_module('.maskdb', inst_lookups)
        mdf_dict = getattr(masks, 'mdf_dict')
        mdf_key = getattr(masks, 'mdf_key')
    except (ImportError, TypeError, AttributeError):
        mdf_dict = None
        mdf_key = None
    else:
        for k, v in mdf_dict.items():
            mdf_dict[k] = path.join(path.dirname(masks.__file__),
                                    'MDF', v)
    caldb = UserDB(name="manual calibrations", mdf_dict=mdf_dict,
                   mdf_key=mdf_key, user_cals=ucals)

    upload_calibs = upload is not None and "calibs" in upload
    upload_science = upload is not None and "science" in upload
    for cls, db, kwargs in parse_databases():
        kwargs["procmode"] = procmode
        if cls == RemoteDB:
            # Actually storing to a remote DB requires that "store" is set in
            # the config *and* the appropriate type is in upload
            kwargs["store_science"] = kwargs["store_cal"] and upload_science
            kwargs["store_cal"] &= upload_calibs
        elif cls == LocalDB:
            kwargs["force_init"] = False
        database = cls(db, name=db, **kwargs)
        caldb.add_database(database)
    return caldb


def parse_databases(default_dbname="cal_manager.db"):
    """
    Parse the databases listed in the global config file. This returns a list
    provided information on how to build the cascase of databases, but does
    not instantiate any CalDB objects, so it can be used by the caldb script
    efficiently.

    Parameters
    ----------
    default_dbname : str
        default name of database file (if only a directory is listed in the
        config file)

    Returns
    -------
    list of tuples (class, database name, kwargs)
    """
    db_list = []
    calconf = get_calconf()
    if not calconf:
        return db_list
    upload_cookie = calconf.get("upload_cookie")
    # Allow old-format file to be read
    try:
        databases = calconf["databases"]
    except KeyError:
        databases = calconf.get("database_dir")
        if not databases:
            return db_list
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn("Use 'databases' instead of 'database_dir' in "
                          "config file.",
                          DeprecationWarning
                          )
    for line in databases.splitlines():
        if not line:  # handle blank lines
            continue
        db, *flags = shlex.split(line)
        # "get" is default if there are no flags, but if any flags are
        # specified, then "get" must be there explicitly
        kwargs = {"get_cal": not bool(flags),
                  "store_cal": False}
        for flag in flags:
            kwarg = f"{flag}_cal"
            if kwarg in kwargs:
                kwargs[kwarg] = True
            else:
                raise ValueError("{}: Unknown flag {!r}".format(db, flag))

        expanded_db = path.expanduser(db)
        if path.isdir(expanded_db):
            db = path.join(db, default_dbname)
            cls = LocalDB
        elif path.isfile(expanded_db):
            cls = LocalDB
        elif "/" in expanded_db and "//" not in expanded_db:
            cls = LocalDB
        else:  # does not check
            cls = RemoteDB
            kwargs["upload_cookie"] = upload_cookie
        db_list.append((cls, db, kwargs))
    return db_list


def set_local_database():
    """
    User helper function to define a local calibration database based on
    the "dragonsrc" config file.

    Returns
    -------
    A LocalDB object
    """
    load_config()
    db_path = get_db_path_from_config()
    db = LocalDB(db_path, log=None)
    return db
