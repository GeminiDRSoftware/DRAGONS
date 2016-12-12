# This parameter file contains the parameters related to the primitives located
# in primitives_calibdb.py, in alphabetical order.

from geminidr import ParametersBASE

class ParametersCalibration(ParametersBASE):
    getCalibration    = {
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
    storeProcessedFringe = {
        "suffix"            : "_fringe",
    }
