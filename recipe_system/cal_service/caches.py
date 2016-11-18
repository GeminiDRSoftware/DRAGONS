#
#                                                                      caches.py
# ------------------------------------------------------------------------------
import os
import pickle

# GLOBAL/CONSTANTS (could be exported to config file)
# [DEFAULT]
CALS = "calibrations"

# [caches]
caches = { 'reducecache'  : '.reducecache',
           'storedcals'   : os.path.join(CALS,"storedcals"),
           'retrievedcals': os.path.join(CALS,"retrievedcals"),
           'calibrations' : CALS
       }

CALDIR     =  caches["storedcals"]
adatadir   = "./recipedata/"                # ???
calindfile = os.path.join('.', caches['reducecache'], "calindex.pkl")
stkindfile = os.path.join('.', caches['reducecache'], "stkindex.pkl")

def set_caches():
    cachedict = {}
    for cachename, cachedir in caches.items():
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        cachedict.update({cachename:cachedir})
    return cachedict

def load_cache(cachefile):
    if os.path.exists(cachefile):
        return pickle.load(open(cachefile, 'r'))
    else:
        return {}

def save_cache(object, cachefile):
    pickle.dump(object, open(cachefile, 'wb'))