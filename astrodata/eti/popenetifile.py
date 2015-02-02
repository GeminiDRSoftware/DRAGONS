from .etifile import ETIFile

class PopenFile(ETIFile):
    name = None
    rc = None
    cmd_frag = None
    def __init__(self, name=None, rc=None):
        print("PopenFile __init__")
        ETIFile.__init__(self, name, rc)
        self.cmd_frag = []

    def get_frag(self):
        print("PopenFile get_frag()")
        return self.cmd_frag

    def prepare(self):
        print("PopenFile prepare()")
        pass

    def recover(self):
        print("PopenFile recover()")

