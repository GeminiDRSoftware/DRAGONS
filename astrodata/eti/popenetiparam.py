from .etiparam import ETIParam

class PopenParam(ETIParam):
    rc = None
    cmd_frag = None
    def __init__(self, rc=None):
        print("PopenParam __init__")
        ETIParam.__init__(self, rc)
        self.cmd_frag = []

    def get_frag(self):
        print("PopenParam get_frag()")
        return self.cmd_frag
    
    def prepare(self):
        print("PopenParam prepare()")
        pass

    def recover(self):
        print("PopenParam recover()")
        pass

    
