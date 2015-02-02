import os
from pyraf import iraf
from astrodata.utils import logutils
from astrodata.eti.pyrafetiparam import PyrafETIParam, IrafStdout

log = logutils.get_logger(__name__)

class GsextractParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gsextract directly.
    """
    rc = None
    key = None
    value = None
    def __init__(self, rc=None, key=None, value=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext

        :param key: A parameter name that is added as a dict key in prepare
        :type key: any

        :param value: A parameter value that is added as a dict value
                      in prepare
        :type value: any
        """
        log.debug("GsextractParam __init__")
        PyrafETIParam.__init__(self, rc)
        self.key = key
        self.value = value

    def nonecheck(self, param=None):
        if param is None or param == "None":
            param = "none"
        return param

    def prepare(self):
        log.debug("Gsextract prepare()")
        self.paramdict.update({self.key:self.value})
        
class FlVardq(GsextractParam):
    rc = None
    fl_vardq = None
    ad = None
    def __init__(self, rc=None, ad=None):
        log.debug("FlVardq __init__")
        GsextractParam.__init__(self, rc)
        if ad.count_exts("VAR") == ad.count_exts("DQ") == ad.count_exts("SCI"):
            self.fl_vardq = iraf.yes
        else:
            self.fl_vardq = iraf.no

    def prepare(self):
        log.debug("FlVardq prepare()")
        self.paramdict.update({"fl_vardq":self.fl_vardq})

class Weights(GsextractParam):
    rc = None
    weights = None
    ad = None
    def __init__(self, rc=None, ad=None):
        log.debug("FlVardq __init__")
        GsextractParam.__init__(self, rc)
        # If fl_vardq=yes, then weights must be "variance"
        if ad.count_exts("VAR") == ad.count_exts("DQ") == ad.count_exts("SCI"):
            self.weights = "variance"
        else:
            self.weights = "none"

    def prepare(self):
        log.debug("Weights prepare()")
        self.paramdict.update({"weights":self.weights})

hardcoded_params = {"Stdout": IrafStdout(),
                    "Stderr":IrafStdout()}

