import numpy as np
import pywcs

from astrodata.utils import Errors
from astrodata.utils import logutils

from gempy.library import astrotools as at
from gempy.gemini import gemini_tools as gt

from .primitives_CORE import PrimitivesCORE

class Resample(PrimitivesCORE):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.

    """
    tagset = set(["GEMINI"])
    
    def init(self, rc):
        PrimitivesCORE.init(self, rc)
        return rc
    init.pt_hide = True
    
    def alignToReferenceFrame(self, adinputs=None, stream='main', **params):
        """
        :param interpolator: type of interpolation desired
        :type interpolator: string, possible values are None, 'nearest', 
                            'linear', 'spline2', 'spline3', 'spline4', 
                            or 'spline5'
        
        :param trim_data: flag to indicate whether output image should be trimmed
                          to the size of the reference image.
        :type trim_data: Boolean
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.alignToReferenceFrame
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
