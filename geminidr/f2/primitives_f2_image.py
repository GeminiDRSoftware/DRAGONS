#
#                                                                  gemini_python
#
#                                                         primitives_f2_image.py
# ------------------------------------------------------------------------------
from gempy.gemini import gemini_tools as gt

from geminidr.core import Image, Photometry
from .primitives_f2 import F2
from . import parameters_f2_image

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class F2Image(F2, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Image level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "F2", "IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super(F2Image, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_f2_image)

    def makeLampFlat(self, adinputs=None, **params):
        """
        This produces an appropriate stacked F2 imaging flat, based on
        the inputs, since one of two procedures must be followed.

        In the standard recipe, the inputs will have come from getList and
        so will all have the same filter and will all need the same recipe.
        """
        log = self.log
        log.debug(gt.log_message("primitive", self.myself(), "starting"))

        # Since this primitive needs a reference, it must no-op without any
        if not adinputs:
            return adinputs

        if adinputs[0].wavelength_band() in ('K',):
            log.stdinfo('Using darkCorrect and stackFlats to make flatfield')
            adinputs = self.darkCorrect(adinputs, dark=params["dark"], do_dark=True)
            stack_params = self._inherit_params(params, "stackFlats")
            stack_params["scale"] = False
            adinputs = self.stackFlats(adinputs, **stack_params)
        else:
            log.stdinfo('Using standard makeLampFlat primitive to make flatfield')
            adinputs = super(F2Image, self).makeLampFlat(adinputs, **params)

        return adinputs
