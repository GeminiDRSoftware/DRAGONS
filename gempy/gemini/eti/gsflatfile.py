import os
import tempfile

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.eti.pyrafetifile import PyrafETIFile
from gempy.gemini import gemini_tools

log = logutils.get_logger(__name__)

class GsflatFile(PyrafETIFile):
    """This class coordinates the ETI files as it pertains to the IRAF
    task gsflat directly.
    """
    rc = None
    diskinlist = None
    pid_str = None
    pid_task = None
    adinput = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GsflatFile __init__")
        PyrafETIFile.__init__(self, rc)
        self.diskinlist = []
        self.taskname = "gsflat"
        self.pid_str = str(os.getpid())
        self.pid_task = self.pid_str + self.taskname
        self.adinput = self.rc.get_inputs_as_astrodata()
    
    def get_prefix(self):
        return "tmp" + self.pid_task

class InAtList(GsflatFile):
    rc = None
    atlist = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("InAtList __init__")
        GsflatFile.__init__(self, rc)
        self.atlist = ""

    def prepare(self):
        log.debug("InAtList prepare()")
        for ad in self.adinput:
            ad = gemini_tools.obsmode_add(ad)
            newname = gemini_tools.filename_updater(adinput=ad, \
                            prefix=self.get_prefix(), strip=True)
            self.diskinlist.append(newname)
            log.fullinfo("Temporary image (%s) on disk for the IRAF task %s" % \
                          (newname, self.taskname))
            ad.write(newname, rename=False, clobber=True)
        self.atlist = "tmpImageList" + self.pid_task
        fh = open(self.atlist, "w")
        for fil in self.diskinlist:
            fh.writelines(fil + "\n")
        fh.close
        log.fullinfo("Temporary list (%s) on disk for the IRAF task %s" % \
                      (self.atlist, self.taskname))
        self.filedict.update({"inflats": "@" + self.atlist})

    def clean(self):
        log.debug("InAtList clean()")
        for file in self.diskinlist:
            os.remove(file)
            log.fullinfo("%s was deleted from disk" % file)
        os.remove(self.atlist)
        log.fullinfo("%s was deleted from disk" % self.atlist)

class OutFile(GsflatFile):
    rc = None
    suffix = None
    recover_name = None
    ad_name = None
    tmp_name = None
    def __init__(self, rc):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("OutFile __init__")
        GsflatFile.__init__(self, rc)
        self.suffix = rc["suffix"]
        self.ad_name = ""
        self.tmp_name = ""

    def prepare(self):
        log.debug("Outfile prepare()")
        outname = gemini_tools.filename_updater(adinput=self.adinput[0], \
                        suffix=self.suffix, strip=True)
        self.ad_name = outname
        self.tmp_name = self.get_prefix() + outname
        self.filedict.update({"specflat": self.tmp_name})

    def recover(self):
        log.debug("OufileETIFile recover()")
        ad = AstroData(self.tmp_name, mode="update")
        ad.filename = self.ad_name
        ad = gemini_tools.obsmode_del(ad)
        log.fullinfo(self.tmp_name + " was loaded into memory")
        return ad

    def clean(self):
        log.debug("Outfile clean()")
        os.remove(self.tmp_name)
        log.fullinfo(self.tmp_name + " was deleted from disk")


class LogFile(GsflatFile):
    rc = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("LogFile __init__")
        GsflatFile.__init__(self, rc)

    def prepare(self):
        log.debug("LogFile prepare()")
        tmplog = tempfile.NamedTemporaryFile()
        self.filedict.update({"logfile": tmplog.name})

