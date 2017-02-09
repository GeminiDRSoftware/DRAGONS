import os
from pyraf import iraf

from gempy.utils import logutils
from gempy.eti_core.pyrafetiparam import PyrafETIParam, IrafStdout
#from gempy.gemini.gemini_tools import calc_nbiascontam

log = logutils.get_logger(__name__)

class GireduceParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gireduce directly.
    """
    inputs = None
    params = None
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
        log.debug("GireduceParam __init__")
        PyrafETIParam.__init__(self, inputs, params)
        self.key = key
        self.value = value

    def nonecheck(self, param=None):
        if param is None or param == "None":
            param = "none"
        return param

    def prepare(self):
        log.debug("Gireduce prepare()")
        self.paramdict.update({self.key:self.value})


class Nbiascontam(GireduceParam):
    inputs = None
    params = None
    nbcontam = None
    ad = None
    def __init__(self, inputs=None, params=None, ad=None):
        log.debug("Nbiascontam __init__")
        GireduceParam.__init__(self, inputs, params)
        osect = self.nonecheck(params["overscan_section"])
        self.nbcontam = 4
        if osect != "none":
            raise NotImplementedError('Cannot use overscan_section.')
            #self.nbcontam = calc_nbiascontam(self.adinput, osect)
        else:
            e2v = False
            if ad:
                detector_type = ad.phu['DETTYPE']
                if detector_type == "SDSU II e2v DD CCD42-90":
                    e2v = True
            else:
                for ad in inputs:
                    detector_type = ad.phu['DETTYPE']
                    if detector_type == "SDSU II e2v DD CCD42-90":
                        e2v = True
            if e2v:
                self.nbcontam = 5
                log.fullinfo("Using default e2vDD nbiascontam = 5")

    def prepare(self):
        log.debug("Nbiascontam prepare()")
        self.paramdict.update({"nbiascontam":self.nbcontam})

subtract_overscan_hardcoded_params = \
                {"gp_outpref"  : "tmp" + str(os.getpid()) + "g",
                 "fl_over"     : iraf.yes,
                 "fl_trim"     : iraf.no,
                 "fl_mult"     : iraf.no,
                 "fl_vardq"    : iraf.no,
                 "fl_bias"     : iraf.no,
                 "fl_flat"     : iraf.no,
                 "mdfdir"     : "",
                 "sat"         : 65000,
                 "gain"        : 2.2000000000000002,
                 "Stdout"      : IrafStdout(),
                 "Stderr"      :IrafStdout()}

