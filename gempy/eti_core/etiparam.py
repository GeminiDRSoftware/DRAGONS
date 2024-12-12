from ..utils import logutils
log = logutils.get_logger(__name__)


class ETIParam:
    inputs = None
    params = None
    def __init__(self, inputs=None, params=None):
        log.debug("ETIParam __init__")
        self.inputs = inputs
        self.params = params

    def prepare(self):
        log.debug("ETIParam prepare()")

    def recover(self):
        log.debug("ETIParam recover()")

    def clean(self):
        log.debug("ETIParam clean(): pass")
