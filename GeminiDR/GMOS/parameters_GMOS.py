# This parameter file contains the parameters related to the primitives located
# in the primitives_GMOS.py file, in alphabetical order.
from GEMINI.parameters_GEMINI import ParametersGemini

class ParametersGMOS(ParametersGemini):
    display = { 
    "extname"           : "SCI",
    "frame"             : 1,
    "ignore"            : False,
    "remove_bias"       : True,
    "threshold"         : "auto",
    "tile"              : True,
    "zscale"            : True,
    }
    measureIQ = {
    "suffix"            : "_iqMeasured",
    "display"           : False,
    "remove_bias"       : True,
    "separate_ext"      : False,
    }
    mosaicDetectors = {
    "suffix"            : "_mosaicked",
    "tile"              : False,
    "interpolate_gaps"  : False,
    "interpolator"      : "linear",
    }
    standardizeInstrumentHeaders = {
    "suffix"            : "_instrumentHeadersStandardized",
    }

    # No default type defined, since the mdf parameter could be a string or
    # an AstroData object
    standardizeStructure = {
    "suffix"            : "_structureStandardized",
    "attach_mdf"        : True,
    "mdf"               : None,
    }

    # No default type defined, since the bias parameter could be a string
    # or an AstroData object
    subtractBias = {
    "suffix"            : "_biasCorrected",
    "bias"            : None,
    }
    subtractOverscan = {
    "suffix"            : "_overscanSubtracted",
    "overscan_section"  : None,
    }
    tileArrays = {
    "suffix"            : "_tiled",
    "tile_all"          : False,
    }
    trimOverscan = {
    "suffix"            : "_overscanTrimmed",
    }
    validateData = {
    "suffix"            : "_dataValidated",
    "repair"            : False,
    }

