#
#                                                          primtives_gmos_mos.py
# ------------------------------------------------------------------------------
import astrodata
import gemini_instruments

from .primitives_gmos_spect import GMOSSpect
from .primitives_gmos_nodandshuffle import GMOSNodAndShuffle
from .parameters_gmos_mos import ParametersGMOSMOS

from recipe_system.utils.decorators import parameter_override
# ------------------------------------------------------------------------------
@parameter_override
class GMOSMOS(GMOSSpect, GMOSNodAndShuffle):
    """
    This is the class containing all of the preprocessing primitives
    for the GMOSLongslit level of the type hierarchy tree. It inherits all
    the primitives from the level above
    """
    tagset = set(["GEMINI", "GMOS", "LONGSLIT"])

    def __init__(self, adinputs, **kwargs):
        super(GMOSMOS, self).__init__(adinputs, **kwargs)
        self.parameters = ParametersGMOSMOS
