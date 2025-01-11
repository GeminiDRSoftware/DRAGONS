from pyraf import iraf
from gempy.utils import logutils
from gempy.eti_core.pyrafetiparam import PyrafETIParam, IrafStdout

log = logutils.get_logger(__name__)

class GemcombineParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gemcombine directly.
    """
    inputs = None
    params = None

    adinput = None
    key = None
    value = None
    def __init__(self, inputs=None, params=None, key=None, value=None):
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
        PyrafETIParam.__init__(self, inputs, params)
        self.adinput = self.inputs
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
    inputs = None
    params = None

    fl_vardq = None
    def __init__(self, inputs=None, params=None):
        log.debug("FlVardq __init__")
        GemcombineParam.__init__(self, inputs, params)
        self.fl_vardq = iraf.no
        for ad in self.adinput:
            if ad.variance is not None:
                self.fl_vardq = iraf.yes
                break

    def prepare(self):
        log.debug("FlVardq prepare()")
        self.paramdict.update({"fl_vardq":self.fl_vardq})

class FlDqprop(GemcombineParam):
    inputs = None
    params = None

    fl_dqprop = None
    def __init__(self, inputs=None, params=None):
        log.debug("FlDqprop __init__")
        GemcombineParam.__init__(self, inputs, params)
        self.fl_dqprop = iraf.no
        for ad in self.adinput:
            if ad.mask is not None:
                self.fl_dqprop = iraf.yes
                break

    def prepare(self):
        log.debug("FlDqprop prepare()")
        self.paramdict.update({"fl_dqprop":self.fl_dqprop})

class Masktype(GemcombineParam):
    inputs = None
    params = None

    masktype = None
    def __init__(self, inputs=None, params=None):
        log.debug("Masktype __init__")
        GemcombineParam.__init__(self, inputs, params)
        if params["apply_dq"]:
            self.masktype = "goodvalue"
        else:
            self.masktype = "none"

    def prepare(self):
        log.debug("Masktype prepare()")
        self.paramdict.update({"masktype":self.masktype})

class Combine(GemcombineParam):
    inputs = None
    params = None
    operation = None
    def __init__(self, inputs=None, params=None):
        log.debug("Combine __init__")
        GemcombineParam.__init__(self, inputs, params)
        self.operation = self.nonecheck(params["operation"])

    def prepare(self):
        log.debug("Combine prepare()")
        self.paramdict.update({"combine":self.operation})

class Nlow(GemcombineParam):
    inputs = None
    params = None
    nlow = None
    def __init__(self, inputs=None, params=None):
        log.debug("Nlow __init__")
        GemcombineParam.__init__(self, inputs, params)
        self.nlow = self.nonecheck(params["nlow"])

    def prepare(self):
        log.debug("Nlow prepare()")
        self.paramdict.update({"nlow":self.nlow})

class Nhigh(GemcombineParam):
    inputs = None
    params = None
    nhigh = None
    def __init__(self, inputs=None, params=None):
        log.debug("Nhigh __init__")
        GemcombineParam.__init__(self, inputs, params)
        self.nhigh = self.nonecheck(params["nhigh"])

    def prepare(self):
        log.debug("Nhigh prepare()")
        self.paramdict.update({"nhigh":self.nhigh})

class Reject(GemcombineParam):
    inputs = None
    params = None
    reject_method = None
    def __init__(self, inputs=None, params=None):
        log.debug("Reject __init__")
        GemcombineParam.__init__(self, inputs, params)
        self.reject_method = self.nonecheck(params["reject_method"])

    def prepare(self):
        log.debug("Reject prepare()")
        self.paramdict.update({"reject":self.reject_method})


hardcoded_params = {'title':'DEFAULT', 'Stdout':IrafStdout(),
                    'Stderr':IrafStdout()}
