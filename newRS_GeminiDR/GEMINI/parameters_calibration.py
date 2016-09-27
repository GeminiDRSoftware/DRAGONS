# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersCalibration(ParametersCORE):
    getCalibration    = {
        "source"            : "all",
        "caltype"           : None,
    }
    storeProcessedArc = {
        "suffix"            : "_arc",
    }
    storeProcessedBias = {
        "suffix"            : "_bias",
    }
    storeProcessedDark = {
        "suffix"            : "_dark",
    } 
    storeProcessedFlat = {
        "suffix"            : "_flat",
    }
