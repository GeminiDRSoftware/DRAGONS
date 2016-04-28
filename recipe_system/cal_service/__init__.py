from ..config import globalConf
from . import prsproxyutil
from . import localmanager

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
            lm = localmanager.LocalManager(calconf.database_dir)
            ret = lm.calibration_search
    except AttributeError:
        # This may happen if either calconf.standalone or
        # calconf.database_dir are not defined
        pass

    return ret
