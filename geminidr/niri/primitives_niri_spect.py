#
#                                                                 gemini_python
#
#                                                      primitives_niri_spect.py
# -----------------------------------------------------------------------------


from importlib import import_module

from geminidr.core import Spect
from gempy.gemini import gemini_tools as gt
from gempy.library import transform, wavecal
from recipe_system.utils.decorators import parameter_override, capture_provenance

from .primitives_niri import NIRI
from . import parameters_niri_spect


@parameter_override
@capture_provenance
class NIRISpect(Spect, NIRI):
    """
    This is the class containing all of the preprocessing primitives for the
    NIRISpect level of the hierarchy tree. It inherits all the primitives from
    the level above.
    """
    tagset = {"GEMINI", "NIRI", "SPECT"}

    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_niri_spect)
