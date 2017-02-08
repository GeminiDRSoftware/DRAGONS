# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos.py file, in alphabetical order.
from geminidr.core.parameters_ccd import ParametersCCD
from geminidr.gemini.parameters_gemini import ParametersGemini

class ParametersGMOS(ParametersGemini, ParametersCCD):
    display = {
        "extname"           : "SCI",
        "frame"             : 1,
        "ignore"            : False,
        "overlay"           : None,
        "remove_bias"       : True,
        "threshold"         : "auto",
        "tile"              : True,
        "zscale"            : True,
    }
    validateData = {
        "suffix"            : "_dataValidated",
        "num_exts"          : [1,2,3,4,6,12],
        "repair"            : False,
    }
