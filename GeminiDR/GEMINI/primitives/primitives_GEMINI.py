# Prototype demo
from astrodata.utils import logutils
from astrodata.utils import Errors

from gempy.gemini import gemini_tools as gt
from gempy.adlibrary.mosaicAD import MosaicAD

from gempy.gemini.gemMosaicFunction import gemini_mosaic_function
from gempy.adlibrary.extract import trace_footprints, cut_footprints

from primitives_bookkeeping import Bookkeeping
from primitives_calibration import Calibration
from primitives_preprocess  import Preprocess
from primitives_register    import Register
from primitives_standardize import Standardize
from primitives_display     import Display

# from primitives_mask import MaskPrimitives
# from primitives_photometry import PhotometryPrimitives
# from primitives_qa import QAPrimitives
# from primitives_resample import ResamplePrimitives
# from primitives_stack import StackPrimitives

from GEMINI.parameters.parameters_GEMINI import ParametersGemini

# ------------------------------------------------------------------------------
class PrimitivesGemini(Standardize, Preprocess, Bookkeeping, Calibration,
                       Register, Display):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    tag = "GEMINI"

    def __init__(self, adinputs):
        super(PrimitivesGemini, self).__init__(adinputs)
        self.parameters = ParametersGemini
    
    def standardizeGeminiHeaders(self, adinputs=None, stream='main', **params):
        """
        Demo prototype.

        """
        self._primitive_exec(self.myself(),
                             self.parameters.standardizeGeminiHeaders,
                             indent=3)

        return
