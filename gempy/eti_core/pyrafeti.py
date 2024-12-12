from .eti import ExternalTaskInterface

from ..utils import logutils
log = logutils.get_logger(__name__)

class PyrafETI(ExternalTaskInterface):
    """This class coordinates the ETI as is relates to all Pyraf tasks"""
    def __init__(self, inputs, params):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("PyrafETI __init__")
        ExternalTaskInterface.__init__(self, inputs, params)

    def execute(self):
        log.debug("PyrafETI.execute()")

    def recover(self):
        log.debug("PyrafETI.recover()")
