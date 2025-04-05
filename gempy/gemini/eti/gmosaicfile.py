import os
import tempfile
import re

import astrodata
import gemini_instruments
from gempy.utils import logutils
from gempy.eti_core.pyrafetifile import PyrafETIFile

from gempy.gemini import gemini_tools

log = logutils.get_logger(__name__)

class GmosaicFile(PyrafETIFile):
    """This class coordinates the ETI files as it pertains to the IRAF
    task gmosaic directly.
    """
    inputs = None
    params = None

    diskinlist = None
    diskoutlist = None
    pid_str = None
    pid_task = None
    adinput = None
    ad = None
    def __init__(self, inputs=None, params=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GmosaicFile __init__")
        PyrafETIFile.__init__(self, inputs, params)
        self.diskinlist = []
        self.diskoutlist = []
        self.taskname = "gmosaic"
        self.pid_str = str(os.getpid())
        self.pid_task = self.pid_str + self.taskname
        if ad:
            self.adinput = [ad]
        else:
            self.adinput = inputs

    def get_prefix(self):
        return "tmp" + self.pid_task

class InAtList(GmosaicFile):
    inputs = None
    params = None
    atlist = None
    ad = None
    def __init__(self, inputs=None, params=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("InAtList __init__")
        GmosaicFile.__init__(self, inputs, params, ad)
        self.atlist = ""

    def prepare(self):
        log.debug("InAtList prepare()")
        for ad in self.adinput:
            ad = gemini_tools.obsmode_add(ad)
            origname = ad.filename
            ad.update_filename(prefix=self.get_prefix(), strip=True)
            self.diskinlist.append(ad.filename)
            log.fullinfo("Temporary image (%s) on disk for the IRAF task %s" % \
                          (ad.filename, self.taskname))
            ad.write(ad.filename, overwrite=True)
            ad.filename = origname
        self.atlist = "tmpImageList" + self.pid_task
        fhdl = open(self.atlist, "w")
        for fil in self.diskinlist:
            fhdl.writelines(fil + "\n")
        fhdl.close()
        log.fullinfo("Temporary list (%s) on disk for the IRAF task %s" % \
                      (self.atlist, self.taskname))
        self.filedict.update({"inimages": "@" + self.atlist})

    def clean(self):
        log.debug("InAtList clean()")
        for a_file in self.diskinlist:
            os.remove(a_file)
            log.fullinfo("%s was deleted from disk" % a_file)
        os.remove(self.atlist)
        log.fullinfo("%s was deleted from disk" % self.atlist)

class OutAtList(GmosaicFile):
    inputs = None
    params = None

    suffix = None
    recover_name = None
    ad_name = None
    atlist = None
    ad = None
    def __init__(self, inputs, params, ad):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("OutAtList __init__")
        GmosaicFile.__init__(self, inputs, params, ad)
        self.suffix = params["suffix"]
        self.ad_name = []
        self.atlist = ""

    def prepare(self):
        log.debug("OutAtList prepare()")
        for ad in self.adinput:
            origname = ad.filename
            ad.update_filename(suffix=self.suffix, strip=True)
            self.ad_name.append(ad.filename)
            self.diskoutlist.append(self.get_prefix() + ad.filename)
            ad.filename = origname
        self.atlist = "tmpOutList" + self.pid_task
        fhdl = open(self.atlist, "w")
        for fil in self.diskoutlist:
            fhdl.writelines(fil + "\n")
        fhdl.close()
        log.fullinfo("Temporary list (%s) on disk for the IRAF task %s" % \
                      (self.atlist, self.taskname))
        self.filedict.update({"outimages": "@" + self.atlist})

    def recover(self):
        log.debug("OutAtList recover()")
        adlist = []
        for i, tmpname in enumerate(self.diskoutlist):
            ad = astrodata.from_file(tmpname)
            ad.filename = self.ad_name[i]
            ad = gemini_tools.obsmode_del(ad)
            adlist.append(ad)
            log.fullinfo(tmpname + " was loaded into memory")
        return adlist

    def clean(self):
        log.debug("OutAtList clean()")
        for tmpname in self.diskoutlist:
            os.remove(tmpname)
            log.fullinfo(tmpname + " was deleted from disk")
        os.remove(self.atlist)
        log.fullinfo(self.atlist + " was deleted from disk")


class LogFile(GmosaicFile):
    inputs = None
    params = None
    def __init__(self, inputs=None, params=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("LogFile __init__")
        GmosaicFile.__init__(self, inputs, params)

    def prepare(self):
        log.debug("LogFile prepare()")
        tmplog = tempfile.NamedTemporaryFile()
        self.filedict.update({"logfile": tmplog.name})

