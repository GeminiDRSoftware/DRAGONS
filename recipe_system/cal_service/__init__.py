import os
from ..config import globalConf, STANDARD_REDUCTION_CONF, DEFAULT_DIRECTORY
from . import transport_request
from . import caches

try:
    from . import localmanager
    localmanager_available = True
except ImportError as e:
    localmanager_available = False
    import_error = str(e)

# BEGIN Setting up the calibs section for config files
CONFIG_SECTION = 'calibs'

globalConf.update_translation({
    (CONFIG_SECTION, 'standalone'): bool
})

globalConf.update_exports({
    CONFIG_SECTION: ('standalone', 'database_dir')
})
# END Setting up the calibs section for config files

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
        # This may happend if there's no calibration config section or, in
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

class Calibrations(dict):
    def __init__(self, calindfile, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self._calindfile = calindfile
        self.update(caches.load_cache(self._calindfile))

    def __getitem__(self, key):
        return self._get_cal(*key)

    def __setitem__(self, key, val):
        self._add_cal(key, val)
        return

    def _add_cal(self, key, val):
        # Munge the key from (ad, caltype) to (ad.calibration_key, caltype)
        key = (key[0].calibration_key(), key[1])
        self.update({key: val})
        caches.save_cache(self, self._calindfile)
        return

    def _get_cal(self, ad, caltype):
        key = (ad.calibration_key(), caltype)
        calfile = self.get(key)
        if calfile is None:
            return None
        else:
            # If the file isn't on disk, delete it from the dict
            if os.path.isfile(calfile):
                return calfile
            else:
                del self[key]
                caches.save_cache(self, self._calindfile)
                return None
