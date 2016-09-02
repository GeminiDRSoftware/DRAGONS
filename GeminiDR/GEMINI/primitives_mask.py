# Prototype demo
from astrodata.utils import logutils
from gempy.gemini import gemini_tools as gt

from .parameters_mask import ParametersMask
from .primitives_CORE import PrimitivesCORE

# ------------------------------------------------------------------------------
class Mask(PrimitivesCORE):
    """
    This is the class containing all of the mask-related primitives
    for the GEMINI level of the type hierarchy tree. It inherits all
    the primitives from the level above, 'GENERALPrimitives'.
    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs):
        super(Mask, self).__init__(adinputs)
        self.parameters = ParametersMask

    def addObjectMaskToDQ(self, rc):
        """
        This primitive combines the object mask in a OBJMASK extension
        into the DQ plane

        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.addToList
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return

    def applyDQPlane(self, rc):
        """
        This primitive sets the value of pixels in the science plane according
        to flags from the DQ plane. 

        :param replace_flags: An integer indicating which DQ plane flags are 
                              to be applied, e.g. a flag of 70 indicates 
                              2 + 4 + 64. The default of 255 flags all values 
                              up to 128.
        :type replace_flags: str

        :param replace_value: Either "median" or "average" to replace the 
                              bad pixels specified by replace_flags with 
                              the median or average of the other pixels, or
                              a numerical value with which to replace the
                              bad pixels. 
        :type replace_value: str
    
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.addToList
        sfx = p_pars["suffix"]
        log.status("-" * len(pmsg))
        log.status(pmsg)
        log.status("-" * len(pmsg))

        timestamp_key = self.timestamp_keys[self.myself()]
        if adinputs:
            self.adinputs = adinputs

        log.stdinfo("Parameters available on {}".format(self.myself()))
        log.stdinfo(str(p_pars))
        log.stdinfo("working on ...")
        for ad in self.adinputs:
            log.stdinfo(ad.filename)

        # Add the appropriate time stamps to the PHU
        gt.mark_history(adinput=ad, primname=self.myself(), keyword=timestamp_key)
        for ad in self.adinputs:
            ad.filename = gt.filename_updater(adinput=ad, suffix=sfx, strip=True)
            log.stdinfo(ad.filename)

        return
