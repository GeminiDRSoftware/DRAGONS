# This parameter file contains the parameters related to the primitives located
# in primitives_calibdb.py, in alphabetical order.

from geminidr import ParametersBASE

class ParametersCalibDB(ParametersBASE):
    getProcessedArc = {
        "refresh"           : True
    }
    getProcessedBias = {
        "refresh"           : True
    }
    getProcessedDark = {
        "refresh"           : True
    }
    getProcessedFlat = {
        "refresh"           : True
    }
    getProcessedFringe = {
        "refresh"           : True
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
