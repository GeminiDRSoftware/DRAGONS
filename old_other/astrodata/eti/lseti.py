from .lsfile import LSFile
from .popeneti import PopenETI
from .lsparam import LSPathParam, LSlafParam


class LSETI(PopenETI):
    def __init__(self, rc):
        print("LSPETI __init__")
        PopenETI.__init__(self, rc)
        self.add_param(LSlafParam(self.rc))
        self.add_param(LSPathParam(self.rc))
        self.cmd_frag = ["ls"]
        inputs = self.rc.get_inputs_as_filenames()
        for inp in inputs:
            self.add_file(LSFile(inp, self.rc))

