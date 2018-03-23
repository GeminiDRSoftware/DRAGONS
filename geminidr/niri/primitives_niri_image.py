#
#                                                                  gemini_python
#
#                                                       primitives_niri_image.py
# ------------------------------------------------------------------------------
from .primitives_niri import NIRI
from ..core import Image, Photometry
from . import parameters_niri_image

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class NIRIImage(NIRI, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the F2Image level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "NIRI", "IMAGE"])

    def __init__(self, adinputs, **kwargs):
        super(NIRIImage, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_niri_image)
