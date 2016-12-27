#
#                                                             primtives_spect.py
# ------------------------------------------------------------------------------
import numpy as np

import astrodata
import gemini_instruments

from .. import PrimitivesBASE
from .parameters_spect import ParametersSpect

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class Spect(PrimitivesBASE):
    """
    This is the class containing all of the preprocessing primitives
    for the Spect level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "SPECT"])

    def __init__(self, adinputs, **kwargs):
        super(Spect, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersSpect