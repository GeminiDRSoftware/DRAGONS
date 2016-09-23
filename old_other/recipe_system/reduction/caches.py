#
#                                                                  gemini_python
#
#                                                                      caches.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
from os.path import join

# GLOBAL/CONSTANTS (could be exported to config file)
# [DEFAULT]
cals = "calibrations"

# [caches]
cachedirs = [".reducecache", cals, 
             join(cals,"storedcals"), 
             join(cals,"retrievedcals")]

CALDIR     =  join(cals,"storedcals")
adatadir   = "./recipedata/"
calindfile = "./.reducecache/calindex.pkl"
stkindfile = "./.reducecache/stkindex.pkl"

#".reducecache/storedcals/storedbiases",
#".reducecache/storedcals/storeddarks",
#".reducecache/storedcals/storedflats",
#".reducecache/storedcals/storedfringes",
#".reducecache/storedcals/retrievedbiases",
#".reducecache/storedcals/retrieveddarks",
#".reducecache/storedcals/retrievedflats",
#".reducecache/storedcals/retrievedfringes",
