from .etiparam import ETIParam

from ..utils import logutils
log = logutils.get_logger(__name__)

class PyrafETIParam(ETIParam):
    rc = None
    paramdict = None
    def __init__(self, rc=None):
        log.debug("PyrafETIParam __init__")
        ETIParam.__init__(self, rc)
        self.paramdict = {}

    def get_parameter(self):
        log.debug("PyrafETIParam get_parameter()")
        return self.paramdict
    
    def prepare(self):
        log.debug("PyrafETIParam prepare()")
        pass

    def recover(self):
        log.debug("PyrafETIParam recover(): pass")
        pass

class IrafStdout():
    """ This is a class to act as the standard output for the IRAF"""
    def __init__(self):
        self.log = log
    
    def flush(self):
        pass
    
    def write(self, out):
        if len(out) > 1:
            self.log.fullinfo(out)

    
