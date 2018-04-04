#
#                                                                  gemini_python
#
#                                                          primtives_gmos_mos.py
# ------------------------------------------------------------------------------
from .primitives_gmos_spect import GMOSSpect
from .primitives_gmos_nodandshuffle import GMOSNodAndShuffle
from . import parameters_gmos_mos

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSMOS(GMOSSpect, GMOSNodAndShuffle):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSLongslit level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS", "SPECT", "MOS"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSMOS, self).__init__(adinputs, **kwargs)
        self._param_update(parameters_gmos_mos)
