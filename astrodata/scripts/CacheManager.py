import os
class CacheMan:
    caches = {  "adcc": ".adcc",
                "adcc.display": ".adcc/display_cache",}
    fileIndex = None
    
    def __init__(self):
        self.fileIndex = {}

    def getCache(self, key):
        if key in self.caches:
            tdir = self.caches[key]
            if not os.path.exists(tdir):
                os.makedirs(tdir)
        return tdir
    def putFile(self, key, filepath):
        self.fileIndex.update({key:filepath})
        
    def getFile(self, key):
        print "CM20:", repr(self.fileIndex)
        return self.fileIndex[key]

cm = CacheMan()
def get_cache_dir(key):
    tdir = cm.getCache(key)
    return tdir

def put_cache_file(key, filepath):
    cm.putFile(key, filepath)
    
def get_cache_file(key):
    return cm.getFile(key)
    
