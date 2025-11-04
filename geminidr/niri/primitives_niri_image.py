#
#                                                                  gemini_python
#
#                                                       primitives_niri_image.py
# ------------------------------------------------------------------------------
from .primitives_niri import NIRI
from ..core import Image, Photometry
from . import parameters_niri_image

import numpy as np

from recipe_system.utils.decorators import parameter_override, capture_provenance


# ------------------------------------------------------------------------------
@parameter_override
@capture_provenance
class NIRIImage(NIRI, Image, Photometry):
    """
    This is the class containing all of the preprocessing primitives
    for the NIRIImage level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = {"GEMINI", "NIRI", "IMAGE"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_niri_image)
