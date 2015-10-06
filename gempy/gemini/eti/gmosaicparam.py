import os
from pyraf import iraf

from astrodata.utils import logutils
from astrodata.eti.pyrafetiparam import PyrafETIParam, IrafStdout
from gempy.gemini.gemini_tools import calc_nbiascontam

log = logutils.get_logger(__name__)

class GmosaicParam(PyrafETIParam):
    """This class coordinates the ETI parameters as it pertains to the IRAF
    task gireduce directly.
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
        log.debug("GmosaicParam __init__")
        PyrafETIParam.__init__(self, rc)
        self.key = key
        self.value = value

    def nonecheck(self, param=None):
        if param is None or param == "None":
            param = "none"
        return param

    def prepare(self):
        log.debug("Gmosaic prepare()")
        self.paramdict.update({self.key:self.value})
        
class FlPaste(GmosaicParam):
    rc = None
    fl_paste = None
    def __init__(self, rc=None):
        log.debug("FlPaste __init__")
        GmosaicParam.__init__(self, rc)
        tile = self.nonecheck(rc["tile"])
        if tile == "none" or tile == False:
            self.fl_paste = iraf.no
        else:
            self.fl_paste = iraf.yes

    def prepare(self):
        log.debug("Flpaste prepare()")
        self.paramdict.update({"fl_paste":self.fl_paste})

class FlFixpix(GmosaicParam):
    rc = None
    fl_fixpix = None
    def __init__(self, rc=None):
        log.debug("FlFixpix __init__")
        GmosaicParam.__init__(self, rc)
        igaps = self.nonecheck(rc["interpolate_gaps"])
        if igaps == "none" or igaps == False:
            self.fl_fixpix = iraf.no
        else:
            self.fl_fixpix = iraf.yes

    def prepare(self):
        log.debug("FlFixpix prepare()")
        self.paramdict.update({"fl_fixpix":self.fl_fixpix})

class Geointer(GmosaicParam):
    rc = None
    geointer = None
    def __init__(self, rc=None):
        log.debug("Geointer __init__")
        GmosaicParam.__init__(self, rc)
        inter = self.nonecheck(rc["interpolator"])
        if inter == "none":
            inter = "linear"
        self.geointer = inter

    def prepare(self):
        log.debug("Geointer prepare()")
        self.paramdict.update({"geointer":self.geointer})

class FlVardq(GmosaicParam):
    rc = None
    fl_vardq = None
    ad = None
    def __init__(self, rc=None, ad=None):
        log.debug("FlVardq __init__")
        GmosaicParam.__init__(self, rc)
        if ad.count_exts("VAR") == ad.count_exts("DQ") == ad.count_exts("SCI"):
            self.fl_vardq = iraf.yes
        else:
            self.fl_vardq = iraf.no

    def prepare(self):
        log.debug("FlVardq prepare()")
        self.paramdict.update({"fl_vardq":self.fl_vardq})

class FlClean(GmosaicParam):
    rc = None
    fl_clean = None
    ad = None
    def __init__(self, rc=None, ad=None):
        log.debug("FlClean __init__")
        GmosaicParam.__init__(self, rc)
        self.fl_clean = iraf.yes

    def prepare(self):
        log.debug("FlClean prepare()")
        self.paramdict.update({"fl_clean":self.fl_clean})

mosaic_detectors_hardcoded_params = { "Stdout"      : IrafStdout(),
                                      "Stderr"      : IrafStdout()}

