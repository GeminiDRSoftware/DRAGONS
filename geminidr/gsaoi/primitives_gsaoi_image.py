#
#                                                                  gemini_python
#
#                                                      primitives_gsaoi_image.py
# ------------------------------------------------------------------------------
from gempy.gemini import gemini_tools as gt

from ..core import Image, Photometry
from .primitives_gsaoi import GSAOI
from .parameters_gsaoi_image import ParametersGSAOIImage

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GSAOIImage(GSAOI, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Image level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GSAOI", "IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super(GSAOIImage, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersGSAOIImage

    def makeLampFlat(self, adinputs=None, **params):
        """
        This produces an appropriate stacked GSAOI imaging flat, based on
        the inputs, since one of two procedures must be followed.

        In the standard recipe, the inputs will have come from getList and
        so will all have the same filter and will all need the same recipe.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Since this primitive needs a reference, it must no-op without any
        if not adinputs:
            return adinputs

        if adinputs[0].wavelength_band() in ['Z', 'J']:
            log.stdinfo('Using stackFrames to make flatfield')
            adinputs = self.stackFrames(adinputs)
        else:
            log.stdinfo('Using lampOnLampOff to make flatfield')
            adinputs = self.lampOnLampOff(adinputs)

        return adinputs