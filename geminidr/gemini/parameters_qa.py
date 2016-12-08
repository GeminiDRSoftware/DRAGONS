# This parameter file contains the parameters related to the primitives located
# in the primitives_qa.py file, in alphabetical order.

from geminidr import ParametersBASE

class ParametersQA(ParametersBASE):
    measureBG = {
        "remove_bias"           : True,
        "separate_ext"          : False,
        "suffix"                : "_bgMeasured",
    }
    measureCC = {
        "suffix"                : "_ccMeasured",
    }
    measureIQ = {
        "display"               : False,
        "remove_bias"           : False,
        "separate_ext"          : False,
        "suffix"                : "_iqMeasured",
    }