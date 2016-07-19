from astrodata.utils import logutils
from astrodata.utils import Errors

from GEMINI.primitives.primitivesGEMINI import PrimitivesGemini

# ------------------------------------------------------------------------------
class PrimitivesImage(PrimitivesGemini):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    tag = "IMAGE"

    # Not sure what to do with this, since it only apply to GMOS and GSAOI.
    # ---------------------------------------------------------------------
    def mosaicADdetectors(self, adinputs=None, stream='main', **params):
        """
        This primitive will mosaic the SCI frames of the input images, along
        with the VAR and DQ frames if they exist.
        
        :param tile: tile images instead of mosaic
        :type tile: Python boolean (True/False), default is False
        
        """
        log = logutils.get_logger(__name__)
        log.debug(gt.log_message("primitive", self.myself(), "starting"))
        pmsg = "{}:{}".format("PRIMITIVE:", self.myself())
        p_pars = self.parameters.mosaicADdetectors
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
    # ---------------------------------------------------------------------
