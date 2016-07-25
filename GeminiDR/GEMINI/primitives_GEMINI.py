# Prototype demo
from .primitives_bookkeeping import Bookkeeping
from .primitives_calibration import Calibration
from .primitives_display     import Display
from .primitives_mask        import Mask
from .primitives_preprocess  import Preprocess
from .primitives_photometry  import Photometry
from .primitives_qa          import QA
from .primitives_stack       import Stack
from .primitives_standardize import Standardize
from .primitives_register    import Register
from .primitives_resample    import Resample

from .parameters_GEMINI import ParametersGemini

# ------------------------------------------------------------------------------
class PrimitivesGemini(Bookkeeping, Calibration, Display, Mask, Preprocess,
                       Photometry, QA, Stack, Standardize, Register, Resample):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    tagset = set(["GEMINI"])

    def __init__(self, adinputs):
        super(PrimitivesGemini, self).__init__(adinputs)
        self.parameters = ParametersGemini
    
    def standardizeGeminiHeaders(self, adinputs=None, stream='main', **params):
        """
        Demo prototype.

        """
        try:
            parset = getattr(self.parameters, self.myself())
        except AttributeError:
            parset = None
        self._primitive_exec(self.myself(), parset=parset, indent=3)
        return
