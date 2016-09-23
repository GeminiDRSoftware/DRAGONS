from .popenetiparam import PopenParam

class LSParam(PopenParam):
    rc = None
    def __init__(self, rc=None):
        print("LSParam __init__")
        PopenParam.__init__(self, rc)

class LSPathParam(LSParam):
    rc = None
    def __init__(self, rc=None):
        print("LSPathParam __init__")
        LSParam.__init__(self, rc)

    def prepare(self):
        print("LSPathParam prepare()")
        if self.rc["lsdir"] is not None:
            self.cmd_frag.append(self.rc["lsdir"])

class LSlafParam(LSParam):
    rc = None
    def __init__(self, rc=None):
        print("LSlafParam __init__")
        LSParam.__init__(self, rc)

    def prepare(self):
        print("LSlafParam prepare()")
        if self.rc["lslaf"] is not None:
            self.cmd_frag.append(self.rc["lslaf"])


