class PrimitivesCatalog:
    catdict = None
    
    def __init__(self):
        self.catdict = {}
        
    def add_primitive_set(self, package, primsetEntry = None, primsetPath = None):
        pdict = {}
        self.catdict.update({primsetEntry : pdict})
        pdict.update({"package":package, "path":primsetPath})
            
    def get_primcat_dict(self, primsetEntry):
        if primsetEntry in self.catdict:
            return self.catdict[primsetEntry]
        else:
            return None
