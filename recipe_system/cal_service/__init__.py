#
#                                                                        DRAGONS
#
#                                                                    cal_service
# ------------------------------------------------------------------------------
import os

from ..config import globalConf, STANDARD_REDUCTION_CONF, DEFAULT_DIRECTORY
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
    Load the configuration from the specified path to file
    (or files), and initialize it with some defaults
    """

    globalConf.load(conf_path,
            defaults = {
                CONFIG_SECTION: {
                    'standalone': False,
                    'database_dir': DEFAULT_DIRECTORY
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
    """

    return (
        localmanager.LocalManager(get_calconf().database_dir).calibration_search
        if is_local() else
        transport_request.calibration_search
    )


def set_calservice(args):
    globalConf.load(STANDARD_REDUCTION_CONF)
    if localmanager_available:
        if args.local_db_dir is not None:
            globalConf.update(CONFIG_SECTION, dict(standalone=True,
                            database_dir=os.path.expanduser(args.local_db_dir)))

    globalConf.export_section(CONFIG_SECTION)
    return
