from __future__ import print_function
#
#                                                                        DRAGONS
#
#                                                                    cal_service
# ------------------------------------------------------------------------------
from os.path import basename
from os.path import expanduser
from os.path import exists

from ..config import globalConf
from ..config import STANDARD_REDUCTION_CONF
from ..config import DEFAULT_DIRECTORY

from . import transport_request

try:
    from . import localmanager
    localmanager_available = True
except ImportError as e:
    localmanager_available = False
    import_error = str(e)

# ------------------------------------------------------------------------------
# BEGIN Setting up the calibs section for config files
CONFIG_SECTION = 'calibs'

globalConf.update_translation({
    (CONFIG_SECTION, 'standalone'): bool
})

globalConf.update_exports({
    CONFIG_SECTION: ('standalone', 'database_dir')
})
# END Setting up the calibs section for config files
# ------------------------------------------------------------------------------
def load_calconf(conf_path=STANDARD_REDUCTION_CONF):
    """
    Load the configuration from the specified path to file (or files), and
    initialize it with some defaults.

    Parameters
    ----------
    conf_path: <str>, Path of configuration file. Default is
                      STANDARD_REDUCTION_CONF -> '~/.geminidr/rsys.cfg'

    Return
    ------
    <ConfigObject>

    """
    globalConf.load(conf_path,
            defaults = {
                CONFIG_SECTION: {
                    'standalone': False,
                    'database_dir': expanduser(DEFAULT_DIRECTORY)
                    }
                })

    return get_calconf()


def update_calconf(items):
    globalConf.update(CONFIG_SECTION, items)


def get_calconf():
    try:
        return globalConf[CONFIG_SECTION]
    except KeyError:
        # This will happen if CONFIG_SECTION has not been defined in any
        # config file, and no defaults have been set (shouldn't happen if
        # the user has called 'load_calconf' before.
        pass


def is_local():
    try:
        if get_calconf().standalone:
            if not localmanager_available:
                raise RuntimeError(
                    "Local calibs manager has been chosen, but there "
                    "are missing dependencies: {}".format(import_error))
            return True

    except AttributeError:
        # This may happen if there's no calibration config section or, in
        # case there is one, if either calconf.standalone or calconf.database_dir
        # are not defined
        pass

    return False


def handle_returns_factory():
    return (
        localmanager.handle_returns
        if is_local() else
        transport_request.handle_returns
    )


def cal_search_factory():
    """
    This function returns the proper calibration search function, depending on
    the user settings.

    Defaults to `prsproxyutil.calibration_search` if there is missing calibs
    setup, or if the `[calibs]`.`standalone` option is turned off.

    Returns
    -------
    calibration_search: <func>
        The appropriate (local or fitsstore) search function indicated by
        a given configuration.

    """

    return (
        localmanager.LocalManager(get_calconf().database_dir).calibration_search
        if is_local() else
        transport_request.calibration_search
    )


def set_calservice(local_db_dir=None, config_file=STANDARD_REDUCTION_CONF):
    """
    Update the calibration service global configuration stored in
    :data:`recipe_system.config.globalConf` by changing the path to the
    configuration file and to the data base directory.

    Parameters
    ----------
    local_db_dir: <str>
        Name of the directory where the database will be stored.

    config_file: <str>
        Name of the configuration file that will be loaded.

    """
    globalConf.load(expanduser(config_file))

    if localmanager_available:
        if local_db_dir is None:
            local_db_dir = globalConf['calibs'].database_dir

        globalConf.update(
            CONFIG_SECTION, dict(
                database_dir=expanduser(local_db_dir),
                config_file=expanduser(config_file)
            )
        )

    globalConf.export_section(CONFIG_SECTION)


class CalibrationService(object):
    """
    The CalibrationService class provides a limited API on the LocalManager
    class. The CalibrationService is meant for public use, as opposed to the
    lower level LocalManager class, and provides limited access to let
    users/callers easy configuration and use of the local calibration database.

    Methods
    -------

    config(db_dir=None, verbose=True, config_file=STANDARD_REDUCTION_CONF)
        configure a session with the database via the rsys.conf file.

    init(wipe=True)
        initialize a calibration database.

    add_cal(path)
        Add a calibration file to the database.

    remove_cal(path)
        Delete a calibration from the database.

    list_files()
        List files in the database. Returns a generator object.

    E.g.,

    >>> from recipe_system.cal_service import CalibrationService
    >>> caldb = CalibrationService()
    >>> caldb.config()
    Using configuration file: ~/.geminidr/rsys.cfg

    The active database directory is:  ~/.geminidr
    The database file to be used: ~/.geminidr/cal_manager.db
    The 'standalone' flag is active; local calibrations will be used.

    >>> caldb.add_cal('calibrations/processed_bias/S20141013S0163_bias.fits')
    >>> for f in caldb.list_files():
            f
    FileData(name='N20120212S0073_flat.fits', path='NIRI/calibrations/processed_flat')
    FileData(name='N20131214S0097_dark.fits', path='NIRI')
    FileData(name='N20150419S0224_flat.fits', path='GMOS_N_TWILIGHT_FLATS')
    FileData(name='S20141013S0020_stackd_flat.fits', path='gband_demo')
    FileData(name='S20141013S0163_bias.fits', path='gband_demo')
    FileData(name='S20141013S0163_flats_bias.fits', path='../gband_demo')
    FileData(name='S20141103S0123_image_bias.fits', path='../gband_demo')

    >>> caldb.remove_cal('N20120212S0073_flat.fits')
    >>> for f in caldb.list_files():
            f
    FileData(name='N20131214S0097_dark.fits', path='NIRI')
    FileData(name='N20150419S0224_flat.fits', path='GMOS_N_TWILIGHT_FLATS')
    FileData(name='S20141013S0020_stackd_flat.fits', path='gband_demo')
    FileData(name='S20141013S0163_bias.fits', path='gband_demo')
    FileData(name='S20141013S0163_flats_bias.fits', path='../gband_demo')
    FileData(name='S20141103S0123_image_bias.fits', path='../gband_demo')

    """
    def __init__(self):
        self.conf = None
        self._mgr = None

    def config(self, db_dir=None, verbose=False,
               config_file=STANDARD_REDUCTION_CONF):
        """
        Configure the Calibration Service and database.

        Parameters
        ----------
        db_dir: <str>
            Path to the local calibration database. If the database has not been
            initialized, call this method and then init() the database. If not
            passed (None), the path specified in a user's rsys.conf file is used.

        verbose: <bool>
            Configuration information will be displayed to stdout.
            Default is True.

        config_file: str
            Path to the configuration file.

        """
        set_calservice(local_db_dir=db_dir, config_file=config_file)
        conf = get_calconf()

        if not conf.standalone:
            print("CalibrationService is not configured as standalone.")

        else:
            print("CalibrationService is configured as standalone.")
            print("The configured local database will be used.")
            self._mgr = localmanager.LocalManager(expanduser(conf.database_dir))

        if verbose:
            self._config_info(conf)

        return

    def init(self, wipe=True):
        """
        Initialize a calibration database. Callers will usually only want to do
        this once. But if called again, will wipe the old database.

        Parameters
        ----------
        wipe: <bool>, optional
            If the database exists and this parameter is `True` (default
            value), the file will be removed and recreated before
            initializing.

        Raises
        ------
        IOError
            If the file exists and there a system error when trying to
            remove it (eg. lack of permissions).

        LocalManagerError
            If the file exists and `wipe` was `False`

        """
        return self._mgr.init_database(wipe=wipe)

    def add_cal(self, path):
        """
        Registers a calibration file specified by 'apath' into the database

        Parameters
        ----------
        path: <str>
            Path to the file. It can be either absolute or relative.

        """
        return self._mgr.ingest_file(path)

    def remove_cal(self, path):
        """
        Removes a calibration file from the database. Note that only the filename
        is relevant. All duplicate copies in the database will be removed.

        Parameters
        ----------
        path: <str>
            Path to the file. It can be either absolute or relative

        """
        return self._mgr.remove_file(basename(path))

    def list_files(self):
        """
        List all files in the local calibration database.

        Parameters
        ----------
        <void>

        Returns
        -------
        LocalManager.list_files: <generator>.
            (See class docstring for example of how to use this generator.)

        Raises
        ------
        LocalManagerError
            Raised when unable to read database.

        """
        return self._mgr.list_files()

    def _config_info(self, conf):
        path = self._mgr._db_path

        is_active = (
            "The 'standalone' flag is \033[1mactive\033[0m; local calibrations"
            "will be used."
        )

        inactive = (
            "The 'standalone' flag is not active; remote calibrations will be"
            " downloaded."
        )

        print()
        print("Using configuration file: \033[1m{}\033[0m".format(conf.config_file))
        print("Active database directory:  \033[1m{}\033[0m".format(conf.database_dir))
        print("Database file: \033[1m{}\033[0m".format(path))
        print()
        print("configuration standalone: {}".format(conf.standalone))

        if conf.standalone:
            print(is_active)
        else:
            print(inactive)

        if not exists(path):
            print("   NB: The database does not exist. Please initialize it.")
            print("   (see init() below.)")
        print()
        return
