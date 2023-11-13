# Defines the LocalDB class for calibration returns. This is a high-level
# interface to the local calibration manager, and provides an API for
# modifying the database on disk.

from os import path, makedirs

from .caldb import CalDB, CalReturn
from .calrequestlib import get_cal_requests, generate_md5_digest

try:
    from . import localmanager
    localmanager_available = True
except ImportError as e:
    localmanager_available = False
    import_error = str(e)

DEFAULT_DB_NAME = "cal_manager.db"


class LocalDB(CalDB):
    """
    The class handling a calibration database stored on disk, via the
    LocalManager class. In addition to the methods required to interface
    with DRAGONS data reduction pipelines, other methods are used to
    provide a full API and effect actions of the "caldb" script.

    An attempt to create an instance of this class without the LocalManager
    being importable will result in an error.

    Attributes
    ----------
    dbfile : str
        name of the file on disk holding the database
    _calmgr : LocalManager instance
        the local calibration manager that will handle the requests
    """
    def __init__(self, dbfile, name=None, valid_caltypes=None, procmode=None,
                 get_cal=True, store_cal=True, log=None, force_init=False):
        if not localmanager_available:
            raise ValueError(f"Cannot initialize local database {name} as"
                             "localmanager could not be imported.\n"
                             f"{import_error}")

        if name is None:  # Do this first so "~" is in the name
            name = dbfile
        dbfile = path.expanduser(dbfile)
        if path.isdir(dbfile) or dbfile.endswith(path.sep):
            dbfile = path.join(dbfile, DEFAULT_DB_NAME)
            name = path.join(name, DEFAULT_DB_NAME)

        super().__init__(name=name, get_cal=get_cal, store_cal=store_cal,
                         log=log, valid_caltypes=valid_caltypes,
                         procmode=procmode)
        self.dbfile = dbfile
        self._calmgr = localmanager.LocalManager(dbfile)
        if not path.exists(dbfile) and force_init:
            self.log.stdinfo(f"Local database file {dbfile} does not exist. "
                             "Initializing.")
            if not path.exists(path.dirname(dbfile)):
                makedirs(path.dirname(dbfile))
            self.init()

    def _get_calibrations(self, adinputs, caltype=None, procmode=None,
                          howmany=1):
        self.log.debug(f"Querying {self.name} for {caltype}")
        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
                                        is_local=True)
        cals = []
        for rq in cal_requests:
            local_cals = self._calmgr.calibration_search(rq, howmany=howmany)
            if not local_cals[0]:
                cals.append(None)
                continue

            good_cals = []
            for calurl, calmd5 in zip(*local_cals):
                calfile = calurl[7:]  # strip "file://"
                cached_md5 = generate_md5_digest(calfile)
                if calmd5 == cached_md5:
                    self.log.debug(f"{rq.filename}: retrieved {calfile}")
                    good_cals.append(calfile)
                else:
                    self.log.warning(f"md5 checksum of {calfile} does not match."
                                     " Not returning this calibration")
            # Append list if >1 requested, else just the filename string
            if good_cals:
                cals.append(good_cals if howmany != 1 else good_cals[0])
            else:
                cals.append(None)

        return CalReturn([None if cal is None else (cal, self.name)
                          for cal in cals])

    def _store_calibration(self, cal, caltype=None):
        """Store the calibration. The LocalDB is not interested in science"""
        if self.store_cal:
            if caltype is None or "science" not in caltype:
                if not path.exists(cal):
                    raise OSError(f"File {cal} does not exist.")
                if caltype is not None:
                    self.log.stdinfo(f"{self.name}: Storing {cal} as {caltype}")
                self._calmgr.ingest_file(cal)
        else:
            self.log.stdinfo(f"{self.name}: NOT storing {cal} as {caltype}")

    # The following methods provide an API to modify the database, by
    # initializing it, removing a named calibration, and listing the files
    # it contains.

    def init(self, wipe=False):
        """
        Initialize a calibration database. Callers will usually only want to do
        this once. But if called again, it will wipe the old database.

        Parameters
        ----------
        wipe : <bool>, optional
            If the database exists and this parameter is `True`, the file will
            be removed and recreated before initializing

        Raises
        ------
        IOError
            If the file exists and there a system error when trying to
            remove it (eg. lack of permissions).

        LocalManagerError
            If the file exists and `wipe` was `False`
        """
        return self._calmgr.init_database(wipe=wipe)

    def add_cal(self, calfile):
        self._store_calibration(calfile)

    def add_directory(self, path, walk=False):
        """
        Ingest one or more files from a given directory, optionally searching
        all subdirectories. This is not used by primitives in the DRAGONS
        data reduction pipelines.

        Parameters
        ----------
        path : str
            directory containing files to be ingested
        walk : bool
            add files from all subdirectories?
        """
        self._calmgr.ingest_directory(path, walk=walk, log=None)

    def remove_cal(self, calfile):
        """
        Removes a calibration file from the database. Note that only the filename
        is relevant. All duplicate copies in the database will be removed.

        Parameters
        ----------
        calfile : <str>
            Path to the file. It can be either absolute or relative
        """
        return self._calmgr.remove_file(path.basename(calfile))

    def list_files(self):
        """
        List all files in the local calibration database. This is not used by
        primitives in the DRAGONS data reduction pipelines.

        Returns
        -------
        LocalManager.list_files: <generator>.
            (See class docstring for example of how to use this generator.)

        Raises
        ------
        LocalManagerError
            Raised when unable to read database.
        """
        return self._calmgr.list_files()
