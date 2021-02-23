# Defines the LocalDB class for calibration returns. This is a high-level
# interface to the local calibration manager, and provides an API for
# modifying the database on disk.

from os import path

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
        if path.isdir(dbfile):
            dbfile = path.join(dbfile, DEFAULT_DB_NAME)

        super().__init__(name=name, get_cal=get_cal, store_cal=store_cal,
                         log=log, valid_caltypes=valid_caltypes,
                         procmode=procmode)
        self.dbfile = dbfile
        self._calmgr = localmanager.LocalManager(dbfile)
        if not path.exists(dbfile) and force_init:
            self.log.stdinfo(f"Local database file {dbfile} does not exist. "
                             "Initializing.")
            self.init()

    def _get_calibrations(self, adinputs, caltype=None, procmode=None):
        self.log.debug(f"Querying {self.name} for {caltype}")
        cal_requests = get_cal_requests(adinputs, caltype, procmode=procmode,
                                        is_local=True)
        cals = []
        for rq in cal_requests:
            # TODO: We can refactor this so it doesn't return lists or URLs,
            # since it no longer has to be the same format as RemoteDB
            calurl, calmd5 = self._calmgr.calibration_search(rq)
            if not calurl:
                cals.append(None)
                continue

            if len(calurl) > 1:
                raise ValueError("Received too many calibrations!")
            calfile, md5 = calurl[0][7:], calmd5[0]  # strip "file://"
            cached_md5 = generate_md5_digest(calfile)
            if md5 == cached_md5:
                self.log.debug(f"{rq.filename}: retrieved {calfile}")
                cals.append(calfile)
            else:
                self.log.warning(f"md5 checksum of {calfile} does not match. "
                                 "Not returning this calibration")
                cals.append(None)

        return CalReturn([None if cal is None else (cal, self.name)
                          for cal in cals])

    def _store_calibration(self, cal, caltype=None):
        """Store the calibration. The LocalDB is not interested in science"""
        if self.store_cal:
            if caltype is None or "science" not in caltype:
                if caltype is not None:
                    self.log.stdinfo(f"{self.name}: Storing {cal} as {caltype}")
                self._calmgr.ingest_file(cal)
        else:
            self.log.stdinfo(f"{self.name}: NOT storing {cal} as {caltype}")

    # The following methods provide an API to modify the database, by
    # initializing it, removing a named calibration, and listing the files
    # it contains.

    def init(self, wipe=True):
        """
        Initialize a calibration database. Callers will usually only want to do
        this once. But if called again, it will wipe the old database.

        Parameters
        ----------
        wipe : <bool>, optional
            If the database exists and this parameter is `True` (default
            value), the file will be removed and recreated before initializing

        Raises
        ------
        IOError
            If the file exists and there a system error when trying to
            remove it (eg. lack of permissions).

        LocalManagerError
            If the file exists and `wipe` was `False`
        """
        return self._calmgr.init_database(wipe=wipe)

    def add_calibration(self, calfile):
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
        self._calmgr.ingest_directory(path, walk=walk, log=self.log)

    def remove_calibration(self, calfile):
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
