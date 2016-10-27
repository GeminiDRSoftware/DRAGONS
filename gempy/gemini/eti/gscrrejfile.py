import os
import tempfile

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.eti.pyrafetifile import PyrafETIFile
from gempy.gemini import gemini_tools

log = logutils.get_logger(__name__)

class GscrrejFile(PyrafETIFile):
    """This class coordinates the ETI files as it pertains to the IRAF
    task gscrrej directly.
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
        log.debug("GscrrejFile __init__")
        PyrafETIFile.__init__(self, rc)
        self.diskinlist = []
        self.diskoutlist = []
        self.taskname = "gscrrej"
        self.pid_str = str(os.getpid())
        self.pid_task = self.pid_str + self.taskname
        if ad:
            self.adinput = [ad]
        else:
            self.adinput = self.rc.get_inputs_as_astrodata()
    
    def get_prefix(self):
        return "tmp" + self.pid_task

class InFile(GscrrejFile):
    rc = None
    ad = None
    diskfile = None
    def __init__(self, rc=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("InFile __init__")
        GscrrejFile.__init__(self, rc, ad)
        self.diskfile = ""

    def prepare(self):
        log.debug("InFile prepare()")
        ad = self.adinput[0]
        ad = gemini_tools.obsmode_add(ad)
        newname = gemini_tools.filename_updater(adinput=ad, \
                            prefix=self.get_prefix(), strip=True)
        self.diskfile = newname
        log.fullinfo("Temporary image (%s) on disk for the IRAF task %s" % \
                         (newname, self.taskname))
        ad.write(newname, rename=False, clobber=True)
        self.filedict.update({"inimage": self.diskfile})

    def clean(self):
        log.debug("InFile clean()")
        os.remove(self.diskfile)
        log.fullinfo("%s was deleted from disk" % self.diskfile)

class OutFile(GscrrejFile):
    rc = None
    suffix = None
    recover_name = None
    ad_name = None
    tmp_name = None
    def __init__(self, rc=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("OutFile __init__")
        GscrrejFile.__init__(self, rc, ad)
        self.suffix = rc["suffix"]
        self.ad_name = ""
        self.tmp_name = ""

    def prepare(self):
        log.debug("Outfile prepare()")
        outname = gemini_tools.filename_updater(adinput=self.adinput[0], \
                        suffix=self.suffix, strip=True)
        self.ad_name = outname
        self.tmp_name = self.get_prefix() + outname
        self.filedict.update({"outimage": self.tmp_name})

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

class LogFile(GscrrejFile):
    rc = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("LogFile __init__")
        GscrrejFile.__init__(self, rc)

    def prepare(self):
        log.debug("LogFile prepare()")
        tmplog = tempfile.NamedTemporaryFile()
        self.filedict.update({"logfile": tmplog.name})

