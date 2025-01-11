from .etifile import ETIFile

from ..utils import logutils
log = logutils.get_logger(__name__)

class PyrafETIFile(ETIFile):
    """This class coordinates the ETI files as it pertains to Pyraf
    tasks in general.
    """
    inputs = None
    params = None
    filedict = None
    def __init__(self, inputs=None, params=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("PyrafETIFile __init__")
        ETIFile.__init__(self, name=None, inputs=inputs, params=params)
        self.filedict = {}

    def get_parameter(self):
        """This returns a parameter as a key value pair to be added
        to a master parameter dict (xcldict) that is used by ETI execute
        """
        log.debug("PyrafETIParam get_parameter()")
        return self.filedict

    def prepare(self):
        log.debug("PyrafETIFile prepare()")

    def recover(self):
        log.debug("PyrafETIFile recover(): pass")

    def clean(self):
        log.debug("PyrafETIFile clean(): pass")
