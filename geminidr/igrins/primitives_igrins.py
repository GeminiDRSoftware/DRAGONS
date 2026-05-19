from geminidr.gemini.primitives_gemini import Gemini
from geminidr.core.primitives_nearIR import NearIR

from . import parameters_igrins

from recipe_system.utils.decorators import parameter_override


@parameter_override
class IGRINS(Gemini, NearIR):
    """
    Top-level primitives for handling GHOST data

    The primitives in this class are applicable to all flavours of GHOST data.
    All other GHOST primitive classes inherit from this class.
    """
    tagset = set()  # Cannot be assigned as a class

    def _initialize(self, adinputs, **kwargs):
        self.inst_lookups = 'geminidr.igrins.lookups'
        super()._initialize(adinputs, **kwargs)
        self._param_update(parameters_igrins)

    def standardizeWCS(self, adinputs=None, **params):
        """
        There is no need at this time to run this primitive on GHOST data.
        No-op.
        """
        return adinputs
