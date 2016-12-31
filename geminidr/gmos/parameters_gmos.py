# This parameter file contains the parameters related to the primitives located
# in the primitives_gmos.py file, in alphabetical order.
from geminidr.core.parameters_ccd import ParametersCCD
from geminidr.gemini.parameters_gemini import ParametersGemini

class ParametersGMOS(ParametersGemini, ParametersCCD):
    validateData = {
        "suffix"            : "_dataValidated",
        "num_exts"          : [1,2,3,4,6,12],
        "repair"            : False,
    }
