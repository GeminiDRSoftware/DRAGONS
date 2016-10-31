from ..utils import logutils
log = logutils.get_logger(__name__)

class ETIFile(object):
    """The base class for all External Class Interface file objects.
    """
    inputs = None
    params = None
    name = None
    def __init__(self, name=None, inputs=None, params=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("ETIFile __init__")
        self.inputs = inputs
        self.params = params
        self.name = name

    def prepare(self):
        print("ETIFile prepare()")

    def recover(self):
        print("ETIFile recover()")

    def clean(self):
        print("ETIFile clean()")
