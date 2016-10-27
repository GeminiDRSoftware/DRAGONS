import os
from pyraf import iraf
from astrodata.utils import logutils
from astrodata.eti.pyrafetiparam import PyrafETIParam, IrafStdout

log = logutils.get_logger(__name__)

class GswavelengthParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gswavelength directly.
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
        log.debug("GswavelengthParam __init__")
        PyrafETIParam.__init__(self, rc)
        self.key = key
        self.value = value

    def nonecheck(self, param=None):
        if param is None or param == "None":
            param = "none"
        return param

    def prepare(self):
        log.debug("Gswavelength prepare()")
        self.paramdict.update({self.key:self.value})
        

class FlInter(GswavelengthParam):
    rc = None
    ad = None
    fl_inter = None
    def __init__(self, rc=None, ad=None):
        log.debug("FlInter __init__")
        GswavelengthParam.__init__(self, rc)
        interactive = self.nonecheck(rc["interactive"])
        if interactive and interactive!="none":
            self.fl_inter = iraf.yes
        else:
            self.fl_inter = iraf.no

    def prepare(self):
        log.debug("FlInter prepare()")
        self.paramdict.update({"fl_inter":self.fl_inter})

hardcoded_params = {"Stdout": IrafStdout(),
                    "Stderr": IrafStdout()}

