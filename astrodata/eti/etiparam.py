

class ETIParam(object):
    rc = None
    def __init__(self, rc=None):
        print ("ETIParam __init__")
        self.rc = rc
    
    def prepare(self):
        print("ETIParam prepare()")
        pass

    def recover(self):
        print("ETIParam recover()")
        pass


