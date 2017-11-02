# This parameter file contains the parameters related to the primitives located
# in the primitives_nearIR.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersNearIR(ParametersBASE):
    addLatencyToDQ = {
        "suffix"        : "_latencyAdded",
        "non_linear"    : False,
        "time"          : 120.,
    }
    makeBPM = {
    }
    lampOnLampOff = {
    }
    separateFlatsDarks = {
    }
    stackDarks = {
    }
    thermalEmissionCorrect = {
    }