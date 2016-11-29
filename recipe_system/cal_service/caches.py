#
#                                                                      caches.py
# ------------------------------------------------------------------------------
import os

# GLOBAL/CONSTANTS (could be exported to config file)
CALS = "calibrations"

# [caches]
caches = {
    'reducecache'  : '.reducecache',
    'calibrations' : CALS
    }

calindfile = os.path.join('.', caches['reducecache'], "calindex.pkl")
stkindfile = os.path.join('.', caches['reducecache'], "stkindex.pkl")

def set_caches():
    cachedict = {}
    for cachename, cachedir in caches.items():
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        cachedict.update({cachename:cachedir})
    return cachedict
