from .etiparam import ETIParam

from ..utils import logutils
log = logutils.get_logger(__name__)

class PyrafETIParam(ETIParam):
    inputs = None
    params = None

    paramdict = None
    def __init__(self, inputs=None, params=None):
        log.debug("PyrafETIParam __init__")
        ETIParam.__init__(self, inputs, params)
        self.paramdict = {}

    def get_parameter(self):
        log.debug("PyrafETIParam get_parameter()")
        return self.paramdict

    def prepare(self):
        log.debug("PyrafETIParam prepare()")

    def recover(self):
        log.debug("PyrafETIParam recover(): pass")

class IrafStdout():
    """ This is a class to act as the standard output for the IRAF"""
    def __init__(self):
        self.log = log

    def flush(self):
        pass

    def write(self, out):
        if len(out) > 1:
            self.log.fullinfo(out)
