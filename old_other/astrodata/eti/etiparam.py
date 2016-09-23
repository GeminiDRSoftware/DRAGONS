from ..utils import logutils
log = logutils.get_logger(__name__)


class ETIParam(object):
    rc = None
    def __init__(self, rc=None):
        log.debug ("ETIParam __init__")
        self.rc = rc
    
    def prepare(self):
        log.debug("ETIParam prepare()")
        pass

    def recover(self):
        log.debug("ETIParam recover()")
        pass

    def clean(self):
        log.debug("ETIParam clean(): pass")
        pass


