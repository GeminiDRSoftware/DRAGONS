#
#                                                                  gemini_python
#
#                                                  astrodata.adutils.reduceutils
#                                                                CacheManager.py
#                                                                   -- DPD Group
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]  # Changed by swapper, 22 May 2014
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
import os

class CacheManager(object):
    def __init__(self):
        self.caches = { "adcc": ".adcc",
                        "adcc.display": ".adcc/display_cache",}
        self.fileIndex = {}

    def getCache(self, key):
        if key in self.caches:
            tdir = self.caches[key]
            if not os.path.exists(tdir):
                os.makedirs(tdir)
        return tdir

    def putFile(self, key, filepath):
        self.fileIndex.update({key:filepath})
        return
        
    def getFile(self, key):
        return self.fileIndex.get(key)

cm = CacheManager()
def get_cache_dir(key):
    tdir = cm.getCache(key)
    return tdir

def put_cache_file(key, filepath):
    cm.putFile(key, filepath)
    return
    
def get_cache_file(key):
    return cm.getFile(key)
    
