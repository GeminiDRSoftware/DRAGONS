from pyraf import iraf
from astrodata.utils import logutils
from astrodata.eti.pyrafetiparam import PyrafETIParam, IrafStdout

log = logutils.get_logger(__name__)

class GemcombineParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gemcombine directly.
    """
    rc = None
    adinput = None
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
        log.debug("GemcombineParam __init__")
        PyrafETIParam.__init__(self, rc)
        self.adinput = self.rc.get_inputs_as_astrodata()
        self.key = key
        self.value = value

    def nonecheck(self, param=None):
        if param is None or param == "None":
            param = "none"
        return param

    def prepare(self):
        log.debug("Gemcombine prepare()")
        self.paramdict.update({self.key:self.value})
        

class FlVardq(GemcombineParam):
    rc = None
    fl_vardq = None
    def __init__(self, rc=None):
        log.debug("FlVardq __init__")
        GemcombineParam.__init__(self, rc)
        self.fl_vardq = iraf.no
        for ad in self.adinput:
            if ad["VAR"]:
                self.fl_vardq = iraf.yes
                break

    def prepare(self):
        log.debug("FlVardq prepare()")
        self.paramdict.update({"fl_vardq":self.fl_vardq})

class FlDqprop(GemcombineParam):
    rc = None
    fl_dqprop = None
    def __init__(self, rc=None):
        log.debug("FlDqprop __init__")
        GemcombineParam.__init__(self, rc)
        self.fl_dqprop = iraf.no
        for ad in self.adinput:
            if ad["DQ"]:
                self.fl_dqprop = iraf.yes
                break

    def prepare(self):
        log.debug("FlDqprop prepare()")
        self.paramdict.update({"fl_dqprop":self.fl_dqprop})

class Masktype(GemcombineParam):
    rc = None
    masktype = None
    def __init__(self, rc=None):
        log.debug("Masktype __init__")
        GemcombineParam.__init__(self, rc)
        if rc["mask"]:
            self.masktype = "goodvalue"
        else:
            self.masktype = "none"

    def prepare(self):
        log.debug("Masktype prepare()")
        self.paramdict.update({"masktype":self.masktype})

class Combine(GemcombineParam):
    rc = None
    operation = None
    def __init__(self, rc=None):
        log.debug("Combine __init__")
        GemcombineParam.__init__(self, rc)
        self.operation = self.nonecheck(rc["operation"])
        
    def prepare(self):
        log.debug("Combine prepare()")
        self.paramdict.update({"combine":self.operation})

class Nlow(GemcombineParam):
    rc = None
    nlow = None
    def __init__(self, rc=None):
        log.debug("Nlow __init__")
        GemcombineParam.__init__(self, rc)
        self.nlow = self.nonecheck(rc["nlow"])
        
    def prepare(self):
        log.debug("Nlow prepare()")
        self.paramdict.update({"nlow":self.nlow})

class Nhigh(GemcombineParam):
    rc = None
    nhigh = None
    def __init__(self, rc=None):
        log.debug("Nhigh __init__")
        GemcombineParam.__init__(self, rc)
        self.nhigh = self.nonecheck(rc["nhigh"])
        
    def prepare(self):
        log.debug("Nhigh prepare()")
        self.paramdict.update({"nhigh":self.nhigh})

class Reject(GemcombineParam):
    rc = None
    reject_method = None
    def __init__(self, rc=None):
        log.debug("Reject __init__")
        GemcombineParam.__init__(self, rc)
        self.reject_method = self.nonecheck(rc["reject_method"])
        
    def prepare(self):
        log.debug("Reject prepare()")
        self.paramdict.update({"reject":self.reject_method})


hardcoded_params = {'title':'DEFAULT','Stdout':IrafStdout(),'Stderr':IrafStdout()}

