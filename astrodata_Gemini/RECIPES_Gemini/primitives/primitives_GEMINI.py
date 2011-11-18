from primitives_GENERAL import GENERALPrimitives
from primitives_bookkeeping import BookkeepingPrimitives
from primitives_display import DisplayPrimitives
from primitives_photometry import PhotometryPrimitives
from primitives_preprocessing import PreprocessingPrimitives
from primitives_qa import QAPrimitives
from primitives_registration import RegistrationPrimitives
from primitives_resample import ResamplePrimitives
from primitives_stack import StackPrimitives
from primitives_standardization import StandardizationPrimitives

class GEMINIPrimitives(BookkeepingPrimitives,DisplayPrimitives,
                       PhotometryPrimitives,PreprocessingPrimitives,
                       QAPrimitives,RegistrationPrimitives,
                       ResamplePrimitives,StackPrimitives,
                       StandardizationPrimitives):
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
