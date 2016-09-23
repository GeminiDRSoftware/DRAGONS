from .popenetifile import PopenFile

class LSFile(PopenFile):
    name = None
    rc = None
    def __init__(self, name=None, rc=None):
        print("LSFile __init__")
        PopenFile.__init__(self, name, rc)

    def prepare(self):
        print "LSFile prepare()"
        self.cmd_frag.append(self.name)
