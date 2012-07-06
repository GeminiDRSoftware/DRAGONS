from primitives_GENERAL import GENERALPrimitives
from primitives_bookkeeping import BookkeepingPrimitives
from primitives_calibration import CalibrationPrimitives
from primitives_display import DisplayPrimitives
from primitives_mask import MaskPrimitives
from primitives_photometry import PhotometryPrimitives
from primitives_preprocess import PreprocessPrimitives
from primitives_qa import QAPrimitives
from primitives_register import RegisterPrimitives
from primitives_resample import ResamplePrimitives
from primitives_stack import StackPrimitives
from primitives_standardize import StandardizePrimitives

class GEMINIPrimitives(BookkeepingPrimitives,CalibrationPrimitives,
                       DisplayPrimitives, MaskPrimitives,
                       PhotometryPrimitives,PreprocessPrimitives,
                       QAPrimitives,RegisterPrimitives,
                       ResamplePrimitives,StackPrimitives,
                       StandardizePrimitives):
    """
    This is the class containing all of the primitives for the GEMINI level of
    the type hierarchy tree. It inherits all the primitives from the level
    above, 'GENERALPrimitives'.
    """
    astrotype = "GEMINI"
    
    def init(self, rc):
        GENERALPrimitives.init(self, rc)
        return rc
    init.pt_hide = True
    
