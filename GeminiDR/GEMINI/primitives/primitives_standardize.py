# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Revision $'[11:-2]
__version_date__ = '$Date $'[7:-2]
# ------------------------------------------------------------------------------
from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from GEMINI.lookups import BPMDict
from GEMINI.lookups import MDFDict

from .primitives_CORE import PrimitivesCORE
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
class Standardize(PrimitivesCORE):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object.

    """
    tag = "GEMINI"
    
    def addDQ(self, adinputs=None, stream='main', **params):
        """
        Demo prototype.

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        timestamp_key = self.timestamp_keys["addDQ"]
        sfx = self.parameters.addDQ["suffix"]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(self.parameters.addDQ))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return


    def addMDF(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        sfx = self.parameters.addMDF["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return


    def addVAR(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        sfx = self.parameters.addVAR["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return
    
    def markAsPrepared(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.markAsPrepared))
        sfx = self.parameters.markAsPrepared["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        logutils.update_indent(0)
        return
    
    def prepare(self):
        """
        Validate and standardize the datasets to ensure compatibility
        with the subsequent primitives.  The outputs, if written to
        disk will be given the suffix "_prepared".
        
        Currently, there are no input parameters associated with 
        this primitive.

        """
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        log = logutils.get_logger(__name__)
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        self.validateData()
        self.standardizeStructure()
        self.standardizeGeminiHeaders()
        self.standardizeInstrumentHeaders()
        self.markAsPrepared()
        logutils.update_indent(0)
        return
