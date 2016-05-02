from ..config import globalConf
from . import prsproxyutil
try:
    from . import localmanager
    localmanager_available = True
except ImportError as e:
    localmanager_available = False
    import_error = e.message

CONFIG_SECTION = 'calibs'

globalConf.update_translation({
    (CONFIG_SECTION, 'standalone'): bool
})

globalConf.update_exports({
    CONFIG_SECTION: ('standalone', 'database_dir')
})

def cal_search_factory():
    calconf = globalConf[CONFIG_SECTION]

    ret = prsproxyutil.calibration_search
    try:
        if calconf.standalone:
            if not localmanager_available:
                raise RuntimeError(
                        "Local calibs manager has been chosen, but there "
                        "are missing dependencies: {}".format(import_error))
            lm = localmanager.LocalManager(calconf.database_dir)
            ret = lm.calibration_search
    except AttributeError:
        # This may happen if either calconf.standalone or
        # calconf.database_dir are not defined
        pass

    return ret
