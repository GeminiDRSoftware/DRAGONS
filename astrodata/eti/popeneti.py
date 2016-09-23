from copy import copy
from subprocess import Popen

from .eti import ExternalTaskInterface

class PopenETI(ExternalTaskInterface):
    cmd_frag = None
    def __init__(self, rc):
        print("PopenETI __init__")
        ExternalTaskInterface.__init__(self, rc)
        self.cmd_frag = []

    def execute(self):
        print("PopenETI.execute()")
        xcmdfrag = copy(self.cmd_frag)
        for par in self.param_dict:
            xcmdfrag.extend(self.param_dict[par].get_frag())
        for fil in self.file_dict:
            xcmdfrag.extend(self.file_dict[fil].get_frag())
        print("\n\nPopen(%s)" % xcmdfrag)
        xproc = Popen(xcmdfrag)
        xproc.wait()
        print("\n\n")
        
