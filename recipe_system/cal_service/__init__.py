import os
from ..config import globalConf
from . import transport_request
from . import caches

try:
    from . import localmanager
    localmanager_available = True
except ImportError as e:
    localmanager_available = False
    import_error = e.message

# BEGIN Setting up the calibs section for config files
CONFIG_SECTION = 'calibs'

globalConf.update_translation({
    (CONFIG_SECTION, 'standalone'): bool
})

globalConf.update_exports({
    CONFIG_SECTION: ('standalone', 'database_dir')
})
# END Setting up the calibs section for config files

def cal_search_factory():
    """
    This function returns the proper calibration search function, depending on
    the user settings.

    Defaults to `prsproxyutil.calibration_search` if there is missing calibs
    setup, or if the `[calibs]`.`standalone` option is turned off.
    """

    ret = transport_request.calibration_search
    try:
        calconf = globalConf[CONFIG_SECTION]
        if calconf.standalone:
            if not localmanager_available:
                raise RuntimeError(
                        "Local calibs manager has been chosen, but there "
                        "are missing dependencies: {}".format(import_error))
            lm = localmanager.LocalManager(calconf.database_dir)
            ret = lm.calibration_search
    except KeyError:
        # This will happen if CONFIG_SECTION has not been defined in any
        # config file
        pass
    except AttributeError:
        # This may happen if either calconf.standalone or
        # calconf.database_dir are not defined
        pass

    return ret

class Calibrations(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        caches.set_caches()
        self.update(caches.load_cache(caches.calindfile))

    def __getitem__(self, key):
        return self._get_cal(*key)

    def __setitem__(self, key, val):
        self._add_cal(key, val)
        return

    def _add_cal(self, key, val):
        # Munge the key from (ad, caltype) to (ad.data_label, caltype)
        key = (key[0].data_label(), key[1])
        self.update({key: val})
        caches.save_cache(self, caches.calindfile)
        return

    def _get_cal(self, ad, caltype):
        key = (ad.data_label(), caltype)
        calfile = self.get(key)
        if calfile is None:
            return None
        else:
            # If the file isn't on disk, delete it from the dict
            if os.path.isfile(calfile):
                return calfile
            else:
                del self[key]
                caches.save_cache(self, caches.calindfile)
                return None
