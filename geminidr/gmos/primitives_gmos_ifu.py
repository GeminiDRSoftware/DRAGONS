#
#                                                          primtives_gmos_ifu.py
# ------------------------------------------------------------------------------
from .primitives_gmos_spect import GMOSSpect
from .primitives_gmos_nodandshuffle import GMOSNodAndShuffle
from . import parameters_gmos_ifu

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSIFU(GMOSSpect, GMOSNodAndShuffle):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSLongslit level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS", "SPECT", "IFU"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSIFU, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_ifu)
