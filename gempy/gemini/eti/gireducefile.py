import os
import tempfile

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.eti.pyrafetifile import PyrafETIFile

from gempy.gemini import gemini_tools

log = logutils.get_logger(__name__)

class GireduceFile(PyrafETIFile):
    """This class coordinates the ETI files as it pertains to the IRAF
    task gireduce directly.
    """
    rc = None
    diskinlist = None
    diskoutlist = None
    pid_str = None
    pid_task = None
    adinput = None
    ad = None
    def __init__(self, rc=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GireduceFile __init__")
        PyrafETIFile.__init__(self, rc)
        self.diskinlist = []
        self.diskoutlist = []
        self.taskname = "gireduce"
        self.pid_str = str(os.getpid())
        self.pid_task = self.pid_str + self.taskname
        if ad:
            self.adinput = [ad]
        else:
            self.adinput = self.rc.get_inputs_as_astrodata()
    
    def get_prefix(self):
        return "tmp" + self.pid_task

class InAtList(GireduceFile):
    rc = None
    atlist = None
    ad = None
    def __init__(self, rc=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("InAtList __init__")
        GireduceFile.__init__(self, rc, ad)
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
        fh.close()
        log.fullinfo("Temporary list (%s) on disk for the IRAF task %s" % \
                      (self.atlist, self.taskname))
        self.filedict.update({"inimages": "@" + self.atlist})

    def clean(self):
        log.debug("InAtList clean()")
        for file in self.diskinlist:
            os.remove(file)
            log.fullinfo("%s was deleted from disk" % file)
        os.remove(self.atlist)
        log.fullinfo("%s was deleted from disk" % self.atlist)

class OutAtList(GireduceFile):
    rc = None
    suffix = None
    recover_name = None
    ad_name = None
    atlist = None
    ad = None
    def __init__(self, rc, ad):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("OutAtList __init__")
        GireduceFile.__init__(self, rc, ad)
        self.suffix = rc["suffix"]
        self.ad_name = []
        self.atlist = ""

    def prepare(self):
        log.debug("OutAtList prepare()")
        for ad in self.adinput:
            outname = gemini_tools.filename_updater(adinput=self.adinput[0], \
                            suffix=self.suffix, strip=True)
            self.ad_name.append(outname)
            self.diskoutlist.append(self.get_prefix() + outname)
        self.atlist = "tmpOutList" + self.pid_task
        fh = open(self.atlist, "w")
        for fil in self.diskoutlist:
            fh.writelines(fil + "\n")
        fh.close()
        log.fullinfo("Temporary list (%s) on disk for the IRAF task %s" % \
                      (self.atlist, self.taskname))
        self.filedict.update({"outimages": "@" + self.atlist})

    def recover(self):
        log.debug("OutAtList recover()")
        adlist = []
        for i, tmpname in enumerate(self.diskoutlist):
            ad = AstroData(tmpname, mode="update")
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


class LogFile(GireduceFile):
    rc = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("LogFile __init__")
        GireduceFile.__init__(self, rc)

    def prepare(self):
        log.debug("LogFile prepare()")
        tmplog = tempfile.NamedTemporaryFile()
        self.filedict.update({"logfile": tmplog.name})

