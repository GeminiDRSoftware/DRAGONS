# This parameter file contains the parameters related to the primitives located
# in the primitives_gemini.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersGemini(ParametersBASE):
    standardizeObservatoryHeaders = {
        "suffix"            : "_observatoryHeadersStandardized",
    }