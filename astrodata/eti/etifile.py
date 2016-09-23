from ..utils import logutils
log = logutils.get_logger(__name__)

class ETIFile(object):
    """The base class for all External Class Interface file objects.
    """
    rc = None
    name = None
    def __init__(self, name=None, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("ETIFile __init__")
        self.rc = rc
        self.name = name
    
    def prepare(self):
        print("ETIFile prepare()")
        pass

    def recover(self):
        print("ETIFile recover()")
        pass
    
    def clean(self):
        print("ETIFile clean()")
        pass
