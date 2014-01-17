#
#                                                                 gemini_python
#
#                                                                      astrodata
#                                                               primitivescat.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision$'[11:-2]
__version_date__ = '$Date$'[7:-2]
# ------------------------------------------------------------------------------

class PrimitivesCatalog(object):
    def __init__(self):
        self.catdict = {}
        
    def add_primitive_set(self, package, primsetEntry = None, primsetPath = None):
        pdict = {}
        self.catdict.update({primsetEntry : pdict})
        pdict.update({"package":package, "path":primsetPath})
        return
            
    def get_primcat_dict(self, primsetEntry):
        if primsetEntry in self.catdict:
            return self.catdict[primsetEntry]
        else:
            return None
