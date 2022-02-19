#
#                                                                 gemini_python
#
#                                                     primitives_gnirs_spect.py
# -----------------------------------------------------------------------------


import astrodata
import gemini_instruments
from gempy.gemini import gemini_tools as gt

from geminidr.gemini.lookups import DQ_definitions as DQ

from .primitives_gnirs import GNIRS
from ..core import NearIR, Spect
from . import parameters_gnirs_longslit

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class GNIRSLongslit(GNIRS, Spect, NearIR):
    """
    This class contains all of the preprocessing primitives for the
    GNIRSLongslit level of the type hierarchy tree. It inherits all the
    primitives from the above level.
    """
    tagset = {'GEMINI', 'GNIRS', 'SPECT'}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_gnirs_longslit)

    def addIllumMaskToDQ(self, adinputs=None, **params):

        pass
