# This parameter file contains the parameters related to the primitives located
# in the primitives_GEMINI.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersResample(ParametersBASE):
    alignToReferenceFrame = {
        "suffix"                : "_align",
        "interpolator"          : "linear",
        "trim_data"             : False,
    }
