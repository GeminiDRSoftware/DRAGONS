from pyraf import iraf
from astrodata.utils import logutils
from astrodata.eti.pyrafetiparam import PyrafETIParam, IrafStdout

log = logutils.get_logger(__name__)

class GsflatParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gsflat directly.
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
        log.debug("GsflatParam __init__")
        PyrafETIParam.__init__(self, rc)
        self.adinput = self.rc.get_inputs_as_astrodata()
        self.key = key
        self.value = value

    def nonecheck(self, param=None):
        if param is None or param == "None":
            param = "none"
        return param

    def prepare(self):
        log.debug("Gsflat prepare()")
        self.paramdict.update({self.key:self.value})
        

class FlVardq(GsflatParam):
    rc = None
    fl_vardq = None
    def __init__(self, rc=None):
        log.debug("FlVardq __init__")
        GsflatParam.__init__(self, rc)

        # Set fl_vardq = yes only if all input have VAR and DQ planes
        for ad in self.adinput:
            if ad["DQ"] and ad["VAR"]:
                self.fl_vardq = iraf.yes
            else:
                self.fl_vardq = iraf.no
                break

    def prepare(self):
        log.debug("FlVardq prepare()")
        self.paramdict.update({"fl_vardq":self.fl_vardq})

hardcoded_params = {'Stdout':IrafStdout(),
                    'Stderr':IrafStdout(),
                    'fl_over':iraf.no,
                    'fl_trim':iraf.no,
                    'fl_bias':iraf.no,
                    'fl_detec':iraf.yes,
                    }

