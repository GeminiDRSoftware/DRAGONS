#
#
#
#                                                        primitives_longslit.py
# -----------------------------------------------------------------------------

from recipe_system.utils.decorators import (parameter_override,
                                            capture_provenance)

from geminidr.core import Spect
from . import parameters_longslit


@parameter_override
@capture_provenance
class Longslit(Spect):
    """This is the class containing primitives specifically for longslit data.
    It inherits all the primitives from the level above.

    Currently this is a placeholder for moving longspit-specific code into in
    the future.

    """
    tagset = {'GEMINI', 'SPECT', 'LS'}
    def _initialize(self, adinputs, **kwargs):
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_longslit)
