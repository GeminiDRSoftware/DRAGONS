# This parameter file imports all generic parameters pertinent to the primitives
# wrapped (inherited) or defined by PrimitivesGemini class in the primitives_GEMINI
# module, in alphabetical order.

from .parameters_bookkeeping import ParametersBookkeeping
from .parameters_calibration import ParametersCalibration
from .parameters_display     import ParametersDisplay
from .parameters_mask        import ParametersMask
from .parameters_preprocess  import ParametersPreprocess
from .parameters_photometry  import ParametersPhotometry
from .parameters_qa          import ParametersQA
from .parameters_register    import ParametersRegister
from .parameters_resample    import ParametersResample
from .parameters_stack       import ParametersStack
from .parameters_standardize import ParametersStandardize

class ParametersGemini(ParametersBookkeeping, ParametersCalibration,
                       ParametersDisplay, ParametersMask, ParametersPreprocess,
                       ParametersPhotometry, ParametersQA, ParametersRegister,
                       ParametersResample, ParametersStack, ParametersStandardize):
    # No default type defined, since the arc parameter could be a string or
    # an AstroData object
    attachWavelengthSolution = {
        "suffix"            : "_wavelengthSolution",
        "arc"               : None,
    }
    cutFootprints = {
        "suffix"            : "_cutfootprints",
    }
    mosaicADdetectors = {
        "suffix"    : "_mosaicAD",
        "tile"      : False,
        # If True, transforms the DQ image bit_plane by bit_plane.
        "dq_planes" : False,
    }
    standardizeGeminiHeaders = {
        "suffix"            : "_geminiHeadersStandardized",
    }
    traceFootprints = {
        "function"          : "polynomial",
        "order"             : 2,
        "trace_threshold"   : 1,                    # enhance_edges function.
        "suffix"            : "_tracefootprints",
    }
