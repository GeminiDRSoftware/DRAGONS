import os
import shutil
import tempfile

from astrodata import AstroData
from astrodata.utils import logutils
from astrodata.eti.pyrafetifile import PyrafETIFile
from gempy.gemini import gemini_tools

log = logutils.get_logger(__name__)

class GswavelengthFile(PyrafETIFile):
    """This class coordinates the ETI files as it pertains to the IRAF
    task gswavelength directly.
    """
    rc = None
    diskinlist = None
    pid_str = None
    pid_task = None
    adinput = None
    ad = None
    def __init__(self, rc=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("GswavelengthFile __init__")
        PyrafETIFile.__init__(self, rc)
        self.diskinlist = []
        self.taskname = "gswavelength"
        self.pid_str = str(os.getpid())
        self.pid_task = self.pid_str + self.taskname
        if ad:
            self.adinput = [ad]
        else:
            self.adinput = self.rc.get_inputs_as_astrodata()
    
    def get_prefix(self):
        return "tmp" + self.pid_task

class InAtList(GswavelengthFile):
    rc = None
    atlist = None
    ad = None
    def __init__(self, rc=None, ad=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("InAtList __init__")
        GswavelengthFile.__init__(self, rc, ad)
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

class OutDatabase(GswavelengthFile):
    rc = None
    ad = None
    suffix = None
    database_name = None
    tmpin_name = None
    recover_name = None
    def __init__(self, rc, ad):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("OutDatabase __init__")
        GswavelengthFile.__init__(self, rc, ad)
        self.suffix = rc["suffix"]
        self.tmpin_name = []
        self.recover_name = []
        self.database_name = ""

    def prepare(self):
        log.debug("OutDatabase prepare()")
        for ad in self.adinput:
            inname = gemini_tools.filename_updater(
                adinput=ad, prefix=self.get_prefix(), strip=True)
            outname = gemini_tools.filename_updater(
                adinput=ad, suffix=self.suffix, strip=True)
            self.tmpin_name.append(inname)
            self.recover_name.append(outname)
        self.database_name = "tmpDatabase" + self.pid_task
        self.filedict.update({"database": self.database_name})

    def recover(self):
        log.debug("OutDatabase recover()")
        adlist = []
        for i, ad in enumerate(self.adinput):
            ad = gemini_tools.obsmode_del(ad)
            ad = gemini_tools.read_database(
                ad, database_name=self.database_name, 
                input_name=self.tmpin_name[i], 
                output_name=ad.phu_get_key_value("ORIGNAME"))
            ad.filename = self.recover_name[i]
            adlist.append(ad)
        return adlist

    def clean(self):
        log.debug("OutDatabase clean()")
        if os.path.exists(self.database_name):
            shutil.rmtree(self.database_name)
        log.fullinfo(self.database_name + " was deleted from disk")


class LogFile(GswavelengthFile):
    rc = None
    def __init__(self, rc=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("LogFile __init__")
        GswavelengthFile.__init__(self, rc)

    def prepare(self):
        log.debug("LogFile prepare()")
        tmplog = tempfile.NamedTemporaryFile()
        self.filedict.update({"logfile": tmplog.name})

