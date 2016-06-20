import os

# Demo prototype primitive sets.
from astrodata.utils import logutils

from gempy.gemini import gemini_tools as gt
# new system imports - 10-06-2016 kra

from GMOS.lookups import GMOSArrayGaps
from GMOS.parameters.parameters_GMOS import ParametersGMOS
from GEMINI.primitives.primitives_GEMINI import PrimitivesGemini

# ------------------------------------------------------------------------------
class PrimitivesGMOS(PrimitivesGemini):
    """
    This is the class containing all of the primitives for the GMOS level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GEMINIPrimitives'.
    """
    tag = "GMOS"

    def __init__(self, adinputs):
        super(PrimitivesGMOS, self).__init__(adinputs)
        self.parameters = ParametersGMOS
    
    def mosaicDetectors(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        self._primitive_exec(self.myself(), self.parameters.mosaicDetectors, indent=3)
        return
 
    def overscanCorrect(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        self.subtractOverscan(params)
        self.trimOverscan(params)
        logutils.update_indent(0)
        return
    
    def standardizeInstrumentHeaders(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        spars = self.parameters.standardizeInstrumentHeaders
        log.stdinfo("Parameters:\n{}".format(spars))
        sfx = spars["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return
    
    def standardizeStructure(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.standardizeStructure))
        sfx = self.parameters.standardizeStructure["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return

    def subtractBias(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.subtractBias))
        sfx = self.parameters.subtractBias["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return
    
    def subtractOverscan(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.subtractOverscan))
        sfx = self.parameters.subtractOverscan["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return

   
    def tileArrays(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.
        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.tileArrays))
        sfx = self.parameters.tileArrays["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return

    def trimOverscan(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.trimOverscan))
        sfx = self.parameters.trimOverscan["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return
    
    def validateData(self, adinputs=None, stream='main', **params):
        """
        This primitive is prototype demo.

        """
        log = logutils.get_logger(__name__)
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        logutils.update_indent(3)
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))
        log.stdinfo("Parameters:\n{}".format(self.parameters.validateData))
        sfx = self.parameters.validateData["suffix"]
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)
        logutils.update_indent(0)
        return

 
