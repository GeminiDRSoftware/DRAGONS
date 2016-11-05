from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from .lookups import BPMDict
from .lookups import MDFDict

from .parameters_standardize import ParametersStandardize
from .primitives_CORE import PrimitivesCORE

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Standardize(PrimitivesCORE):
    """
    This is the class containing all of the primitives used to standardize an
    AstroData object.

    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs, context, ucals=None, uparms=None):
        super(Standardize, self).__init__(adinputs, context, ucals=ucals, uparms=uparms)
        self.parameters = ParametersStandardize


    def addDQ(self, adinputs=None, stream='main', **params):
        """
        Demo prototype.

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        timestamp_key = self.timestamp_keys["addDQ"]
        sfx = self.parameters.addDQ["suffix"]
        log.stdinfo("timestamp keyword: {}".format(timestamp_key))
        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(self.parameters.addDQ))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return

    def addMDF(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        sfx = self.parameters.addMDF["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return

    def addVAR(self, adinputs=None, stream='main', **params):
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        sfx = self.parameters.addVAR["suffix"]
        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(self.parameters.addVAR))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return
    
    def markAsPrepared(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.markAsPrepared))
        sfx = self.parameters.markAsPrepared["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

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
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        self.validateData()
        self.standardizeStructure()
        self.standardizeGeminiHeaders()
        self.standardizeInstrumentHeaders()
        self.markAsPrepared()
        return
