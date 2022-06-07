#
#                                                                 gemini_python
#
#                                                     primitives_f2_longslit.py
# -----------------------------------------------------------------------------

from . import parameters_f2_longslit
from .primitives_f2_spect import F2Spect

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

# -----------------------------------------------------------------------------
@parameter_override
@capture_provenance
class F2Longslit(F2Spect):
    """This class contains all of the processing primitives for the F2Longslit
    level of the type hiearchy tree. It inherits all the primitives from the
    above level.
    """
    tagset = {'GEMINI', 'F2', 'SPECT', 'LS'}
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_f2_longslit)
