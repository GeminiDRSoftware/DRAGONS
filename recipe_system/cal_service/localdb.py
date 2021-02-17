# Defines the LocalDB class for calibration returns. This is a high-level
# interface to the local calibration manager, and provides an API for
# modifying the database on disk.

from os.path import basename, expanduser

from .caldb import CalDB, CalReturn
from .localmanager import LocalManager
from .calrequestlib import get_cal_requests, generate_md5_digest


class LocalDB(CalDB):
    def __init__(self, dbfile, name=None, valid_caltypes=None,
                 get=True, store=True, log=None):
        if name is None:
            name = dbfile
        super().__init__(name=name or dbfile, get=get, store=store, log=log,
                         valid_caltypes=valid_caltypes)
        self._calmgr = LocalManager(expanduser(dbfile))

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
        if self.store:
            if caltype and "science" not in caltype:
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

    def remove_calibration(self, calfile):
        """
        Removes a calibration file from the database. Note that only the filename
        is relevant. All duplicate copies in the database will be removed.

        Parameters
        ----------
        calfile : <str>
            Path to the file. It can be either absolute or relative
        """
        return self._calmgr.remove_file(basename(calfile))

    def list_files(self):
        """
        List all files in the local calibration database.

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
