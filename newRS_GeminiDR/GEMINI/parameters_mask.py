# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from parameters_CORE import ParametersCORE

class ParametersMask(ParametersCORE):
    addObjectMaskToDQ = {
        "suffix"    : "_objectMaskAdded"
    }
    applyDQPlane = {
        "suffix"        : "_dqPlaneApplied",
        "replace_flags" : 255,
        "replace_value" : "median",
    }
