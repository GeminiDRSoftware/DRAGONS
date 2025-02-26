#
#                                                                        DRAGONS
#
#                                                                localmanager.py
# ------------------------------------------------------------------------------

import os
from os.path import abspath, basename, dirname

import warnings
from collections import namedtuple

from sqlalchemy.exc import SAWarning, OperationalError

from fits_storage import gemini_metadata_utils as fsgmu
from fits_storage.config import get_config
from fits_storage import db
from fits_storage.db.selection import Selection
from fits_storage.db.list_headers import list_headers
from fits_storage.db.createtables import create_tables
from fits_storage.db.remove_file import remove_file
from fits_storage.core.ingester import Ingester
from fits_storage.queues.orm.ingestqueueentry import IngestQueueEntry
from fits_storage.cal.calibration import get_cal_object

from gempy.utils import logutils

# ------------------------------------------------------------------------------
__all__ = ['LocalManager', 'LocalManagerError']
# ------------------------------------------------------------------------------
log = logutils.get_logger(__name__)
# ------------------------------------------------------------------------------
# SQLAlchemy complains about SQLite details. We can't do anything about the
# data types involved, because the ORM files are meant for PostgreSQL.
# The queries work, though, so we just ignore the warnings
warnings.filterwarnings(
    'ignore',
    r"Dialect sqlite\+pysqlite does \*not\* support Decimal objects natively, "
    r"and SQLAlchemy must convert from floating point - rounding errors and "
    r"other issues may occur. Please consider storing Decimal numbers as "
    r"strings or integers on this platform for lossless storage\.",
    SAWarning, r'^.*$'
)

extra_descript = {
    'GMOS_NODANDSHUFFLE': 'nodandshuffle',
    'SPECT': 'spectroscopy',
    'OVERSCAN_SUBTRACTED': 'overscan_subtracted',
    'OVERSCAN_TRIMMED': 'overscan_trimmed',
    'PREPARED': 'prepared'
}

args_for_cals = {
    # cal_type      : (method_name, {arg_name: value, ...})
    'processed_arc':  ('arc', {'processed': True}),
    'processed_bias': ('bias', {'processed': True}),
    'processed_dark': ('dark', {'processed': True}),
    'processed_flat': ('flat', {'processed': True}),
    'processed_pinhole': ('pinhole', {'processed': True}),
    'processed_standard': ('standard', {'processed': True}),
    'processed_slitillum': ('slitillum', {'processed': True}),
    'processed_bpm': ('bpm', {'processed': True})
}

ERROR_CANT_WIPE = 0
ERROR_CANT_CREATE = 1
ERROR_CANT_READ = 2
ERROR_DIDNT_FIND = 3
ERROR_MISSING_DATABASE_FILE = 4

FileData = namedtuple('FileData', 'name path')


class LocalManagerError(Exception):
    def __init__(self, error_type, message, *args, **kw):
        super().__init__(message, *args)
        self.message = message
        self.error_type = error_type


def ensure_db_file(func):
    """
    Decorator for functions in
    :class:`~recipe_system.cal_service.localmanager.LocalManager`
    that we want to require the database file exist for.  If we don't check,
    SQLAlchemy will just silently create the DB file.

    Parameters
    ----------
    func : function
        Function to decorate

    Returns
    -------
    function : decorator call
    """

    def wrapper_ensure_db_file(self, *args, **kwargs):
        if self.path != ":memory:":
            if not os.path.exists(self.path):
                raise LocalManagerError(ERROR_MISSING_DATABASE_FILE,
                                        f"Unable to find calibration database file {self.path}")
            if os.path.isdir(self.path):
                raise LocalManagerError(ERROR_MISSING_DATABASE_FILE,
                                        f"Calibration database file {self.path} is a directory.  It should be a file")
        return func(self, *args, **kwargs)
    return wrapper_ensure_db_file


class LocalManager:
    def __init__(self, db_path):
        self._db_path = db_path
        self.fsc = None  # FitsStorage config object
        self.session = None  # FitsStorage database session
        self.ingester = None  # FitsStorage file ingester. Lazy instantiated
        self._reset()

    @property
    def path(self):
        return self._db_path

    def _reset(self):
        """
        Configures fits_storage to run within DRAGONS, then sets a new database
        session object for this instance.
        """

        configstring = f"""
        [DEFAULT]
        fits_server_name: DRAGONS embedded local calibration manager
        storage_root: {abspath(dirname(self._db_path))}  
        database_url: {'sqlite:///' + self._db_path}      
        """

        self.fsc = get_config(configstring=configstring, builtinonly=True,
                              reload=True)

        log.debug(f"FitsStorage config database_url: {self.fsc.database_url}")
        self.session = db.sessionfactory(reload=True)

    def init_database(self, wipe=True):
        """
        Initializes a SQLite database with the tables required for the
        calibration manager.

        Parameters
        ----------
        wipe: bool, optional
            If the database exists and this parameter is `True` (default
            value), the file will be removed and recreated before
            initializing.

        Raises
        ------
        IOError
            If the file exists and there a system error when trying to
            remove it (e.g. lack of permissions).

        LocalManagerError
            If the file exists and `wipe` was `False`
        """

        if self.fsc is None:
            raise LocalManagerError(ERROR_CANT_CREATE,
                                    "FitsStorage has not been configured")

        # Trim the 'sqlite:///' from the start of the database_url.
        # The 4th / if present is the start of the path
        if self.fsc.database_url.startswith('sqlite:///'):
            db_path = self.fsc.database_url[10:]
        else:
            raise LocalManagerError(ERROR_CANT_WIPE,
                                    "Non SQLite database URL. This is not "
                                    "currently supported within DRAGONS")

        if os.path.exists(db_path) and wipe:
            try:
                os.remove(db_path)
            except Exception:
                raise LocalManagerError(ERROR_CANT_WIPE,
                                        "Failed to remove {db_path}")


        try:
            log.debug(f"Creating Tables with dburl: {self.fsc.database_url}")
            create_tables(self.session)
            self.session.commit()
        except OperationalError:
            message = (f"There was an error when trying to create the database "
                       f"{db_path}. Please, check your path and permissions.")
            raise LocalManagerError(ERROR_CANT_CREATE, message)

    @ensure_db_file
    def remove_file(self, path):
        """
        Removes a file from the database. Note that only the filename
        is relevant. All duplicate copies in the database will be
        removed

        Parameters
        ----------
        path: string
            Path to the file. It can be either absolute or relative
        """
        remove_file(path, self.session)

    @ensure_db_file
    def ingest_file(self, path):
        """
        Registers a file into the database

        Parameters
        ----------
        path: str
            Path to the file. It can be either absolute or relative
        """
        directory = abspath(dirname(path))
        filename = basename(path)

        if self.ingester is None:
            # We pass the DRAGONS logger here in place of a fits storage logger
            self.ingester = Ingester(self.session, log,
                                     override_engineering=True)

        try:
            # Make an IngestQueueEntry for the file to ingest. We could just
            # make a mock-up dictionary instead, but there are some defaults and
            # path manipulations in the IQE class that are convenient to have.
            # We never add this object to the database session; ingester
            # updates it and commits the session, that will basically be a noop
            # because the iqe isn't part of the session.
            iqe = IngestQueueEntry(filename, directory)
            self.ingester.ingest_file(iqe)

            if iqe.failed:
                log.warn(f"Ingest failed for file {iqe.filename}: {iqe.error}")

        except Exception as err:
            self.session.rollback()
            self.remove_file(path)
            raise err

    def ingest_directory(self, path, walk=False, log=None):
        """Registers into the database all FITS files under a directory

        Parameters
        ----------
        path: <str>, optional
            Path to the root directory. It can be either absolute or
            relative.

        walk: <bool>, optional
            If `False` (default), only the files in the top level of the
            directory will be considered. If `True`, all the subdirectories
            under the path will be explored in search of FITS files.

        log: <function>, optional
            If provided, it must be a function that accepts a single argument,
            a message string. This function can then process the message
            and log it into the proper place.

        """
        for root, dirs, files in os.walk(path):
            for fname in [l for l in files if l.endswith('.fits')]:
                self.ingest_file(os.path.join(root, fname))
                if log:
                    log("Ingested {}/{}".format(root, fname))

    @ensure_db_file
    def calibration_search(self, rq, howmany=1):
        """
        Performs a search in the database using the requested criteria.

        Parameters
        ----------
        rq: <instance>, CalibrationRequest
            Contains search criteria, including instrument, descriptors, etc.

        howmany: <int>
            Maximum number of calibrations to return


        Returns
        -------
        result: <tuple>
            A tuple of exactly two elements.

            In the case of success, the tuple contains two lists, the first
            being the URLs to calibration files, and the second the MD5 sums.

            When an error occurs, the first element in the tuple will be
            `None`, and the second a string describing the error.

        """
        caltype = rq.caltype
        descripts = rq.descriptors
        types = rq.tags

        if "ut_datetime" in descripts:
            utc = descripts["ut_datetime"]
            descripts.update({"ut_datetime":utc})

        for (type_, desc) in list(extra_descript.items()):
            descripts[desc] = type_ in types

        # Obtain a calibration manager object instantiated according to the
        # instrument.
        cal_obj = get_cal_object(self.session, filename=None, header=None,
                                 descriptors=descripts, types=types,
                                 procmode=rq.procmode)

        caltypes = fsgmu.cal_types if caltype == '' else [caltype]

        # The function that downloads an XML returns only the first result,
        # so we won't bother iterating over the whole thing in case that
        # caltype was empty.
        ct = caltypes[0]
        method, args = args_for_cals.get(ct, (ct, {}))

        # Obtain a list of calibrations for the specified cal type
        cals = getattr(cal_obj, method)(**args)

        ret_value = []
        for cal in cals:
            if cal.diskfile.present and len(ret_value) < howmany:
                ret_value.append(('file://{}'.format(cal.diskfile.fullpath),
                                  cal.diskfile.data_md5))

        if ret_value:
            # Turn from list of tuples into two lists
            return tuple(map(list, list(zip(*ret_value))))

        return None, "Could not find a proper calibration in the local database"

    @ensure_db_file
    def list_files(self):
        selection = Selection({'canonical': True})
        orderby = 'filename'

        try:
            headers = list_headers(selection, orderby, self.session)
        except OperationalError:
            message = "There was an error when trying to read from the database."
            raise LocalManagerError(ERROR_CANT_READ, message)

        for header in headers:
            yield FileData(header.diskfile.filename, header.diskfile.path)
