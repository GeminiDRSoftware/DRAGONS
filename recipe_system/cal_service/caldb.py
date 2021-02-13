import abc
import os
from functools import partial

from gempy.utils import logutils

# Subdirectory in which to store processed calibrations
CALDIR = "calibrations"

REQUIRED_TAG_DICT = {'processed_arc': ['PROCESSED', 'ARC'],
                     'processed_bias': ['PROCESSED', 'BIAS'],
                     'processed_dark': ['PROCESSED', 'DARK'],
                     'processed_flat': ['PROCESSED', 'FLAT'],
                     'processed_fringe': ['PROCESSED', 'FRINGE'],
                     'processed_standard': ['PROCESSED', 'STANDARD'],
                     'processed_slitillum': ['PROCESSED', 'SLITILLUM'],
                     'bpm': ['BPM'],
                     }

VALID_CALTYPES = REQUIRED_TAG_DICT.keys()

def cascade(fn):
    """
    Decorates methods such that, after execution, the method with the same
    name is called on the next CalDB instance in the cascade, with the same
    arguments.
    """
    def wrapper(instance, *args, **kwargs):
        fn(instance, *args, **kwargs)
        if instance.nextdb is not None:
            getattr(instance.nextdb, fn.__name__)(*args, **kwargs)
    return wrapper


class CalDB(metaclass=abc.ABCMeta):
    def __init__(self, name=None, get=True, store=False,
                 valid_caltypes=None, log=None):
        self._valid_caltypes = valid_caltypes or VALID_CALTYPES
        self.caldir = CALDIR
        self.name = name
        self.get = get
        self.store = store
        self.nextdb = None
        self.log = log or logutils.get_logger(__name__)

    def __getattr__(self, attr):
        """
        A shortcut to avoid having to define every single get_processed_cheese
        method to simply call get_calibrations(caltype="processed_cheese")
        """
        if (attr.startswith("get_processed_") and
                attr[4:] in self._valid_caltypes):
            return partial(self.get_calibrations, caltype=attr[4:])
        raise AttributeError("Unknown attribute {!r}".format(attr))

    def __len__(self):
        """Return the number of separate databases"""
        return 1 if self.nextdb is None else 1 + len(self.nextdb)

    def add_database(self, db):
        """
        Adds a database to the end of the cascade. It does this recursively,
        calling this method on the next database in the casecade, until it
        gets to the end.

        Parameters
        ----------
        db : CalDB instance
            the database to add to the end of this cascade
        """
        if self.nextdb is None:
            self.nextdb = db
        else:
            self.nextdb.add_database(db)

    def get_calibrations(self, adinputs, caltype=None, procmode=None):
        """
        Returns the requested calibration type for the list of files provided.

        Parameters
        ----------
        adinputs : list of AstroData objects
            the files requiring calibrations
        caltype : str
            the type of calibration required
        procmode : str/None
            minimum quality of processing needed for the calibration

        Returns
        -------
        CalReturn : the calibration filenames and the databases which
                    provided them
        """
        if caltype not in self._valid_caltypes:
            raise ValueError("Calibration type {!r} not recognized".
                             format(caltype))
        if self.get:
            cal_ret = self._get_calibrations(adinputs, caltype=caltype,
                                             procmode=procmode)
        else:
            cal_ret = CalReturn([None] * len(adinputs))

        if cal_ret.files.count(None) and self.nextdb:
            # Pass adinputs without calibrations to next database (with
            # possible recursion)
            new_ret = self.nextdb.get_calibrations(
                [ad for ad, cal in zip(adinputs, cal_ret.files) if cal is None],
                caltype=caltype, procmode=procmode)
            # Insert any new calibrations into the original list
            new_files, new_origins = iter(new_ret.files), iter(new_ret.origins)
            return CalReturn([next(new_files) if cal is None else cal
                              for cal in cal_ret.files],
                             [next(new_origins) if origin is None else origin
                              for origin in cal_ret.origins])
        return cal_ret

    @cascade
    def set_calibrations(self, adinputs, caltype=None, calfile=None):
        """
        Manually assign files as calibrations of a specified type to a list
        of one or more AstroData objects.

        Parameters
        ----------
        adinputs: list of AstroData objects
            the files to be assigned calibrations
        caltype : str
            the type of calibration being assigned
        calfile : str
            the filename of the calibration
        """
        self._set_calibrations(adinputs, caltype=caltype, calfile=calfile)

    @cascade
    def unset_calibrations(self, adinputs, caltype=None):
        """
        Manually remove calibrations of a specified type from a list
        of one or more AstroData objects.

        Parameters
        ----------
        adinputs: list of AstroData objects
            the files to have calibrations removed
        caltype : str
            the type of calibration to be removed
        """
        self._unset_calibrations(adinputs, caltype=caltype)

    @cascade
    def clear_calibrations(self):
        """
        Clear all manually-assigned calibrations.
        """
        self._clear_calibrations()

    def store_calibration(self, cal, caltype=None):
        """
        Handle storage of calibrations. For all types of calibration
        (NOT processed_science) this saves the file to disk in the
        appropriate calibrations/ subdirectory (if store=True) and
        then passes the filename down the chain. For processed_science
        files, it passes the AstroData object.

        Parameters
        ----------
        cals : str/AstroData
            the calibration or its filename
        caltype : str
            type of calibration
        """
        # Create storage directory if it doesn't exist and we've got an AD
        # object. This will only happen with the first CalDB to store,
        # since the calibration will be forwarded as a filename.
        if not (isinstance(cal, str) or "science" in caltype):
            if not os.path.exists(os.path.join(self.caldir, caltype)):
                os.makedirs(os.path.join(self.caldir, caltype))

            required_tags = REQUIRED_TAG_DICT[caltype]
            if cal.tags.issuperset(required_tags):
                fname = os.path.join(self.caldir, caltype,
                                     os.path.basename(cal.filename))
                cal.write(fname, overwrite=True)
            else:
                self.log.warning(
                    f"File {cal.filename} is not recognized as a"
                    f" {caltype}. Not storing as a calibration.")
                return
            cal = fname

        self._store_calibration(cal, caltype=caltype)
        if self.nextdb is not None:
            self.nextdb.store_calibration(cal, caltype=caltype)

    @abc.abstractmethod
    def _get_calibrations(self, adinputs, caltype=None, procmode=None):
        pass

    @abc.abstractmethod
    def _store_calibration(self, cals, caltype=None):
        pass

    def _set_calibrations(self, adinputs, caltype=None, calfile=None):
        pass

    def _unset_calibrations(self, adinputs, caltype=None):
        pass

    def _clear_calibrations(self):
        pass


class CalReturn:
    """
    A simple class to store files and origins (i.e., which CalDB instance
    provided the calibration) for calibration requests. The class can be
    instantiated with either a list of files and a list of origins, or a
    single list of 2-tuples (file, origin), where a single None can be
    used in place of (None, None).

    Public attributes are:
    files : list of files
    origins : list of origins

    There is also an items() method that returns these lists as zipped pairs,
    like the dict.items() method.
    """
    def __init__(self, *args):
        if len(args) == 1:
            self.files = [item[0] if item else None for item in args[0]]
            self.origins = [item[1] if item else None for item in args[0]]
        elif len(args) == 2:
            self.files, self.origins = args
        else:
            raise ValueError("CalReturn must be initialized with 1 or 2 "
                             "arguments")

    def items(self):
        return list((file, origin) for file, origin in zip(self.files,
                                                           self.origins))
