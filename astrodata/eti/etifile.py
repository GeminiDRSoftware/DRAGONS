

class ETIFile(object):
    rc = None
    name = None
    def __init__(self, name=None, rc=None):
        print("ETIFile __init__")
        self.rc = rc
        self.name = name
    
    def prepare(self):
        print("ETIFile prepare()")
        pass

    def recover(self):
        print("ETIFile recover()")
        pass
